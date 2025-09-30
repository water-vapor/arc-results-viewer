import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder

from utils import grid_to_image, grids_equal


st.set_page_config(page_title="ARC Results Viewer", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def load_df(parquet_path: str | None) -> pd.DataFrame:
    path = parquet_path or "arc_results_processed.parquet"
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        st.warning("No data found. Run preprocess_data.py first.")
        return pd.DataFrame()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return pd.DataFrame()

    df["problem_id"] = df["problem_id"].astype(str)
    if "assets" not in df.columns:
        df["assets"] = [{} for _ in range(len(df))]
    return df


@st.cache_data(show_spinner=False)
def load_arc_dataset() -> tuple[Dict[str, Any], Dict[str, List[list]]]:
    """Return the ARC training challenges + solutions for context grids."""

    try:
        with open("dataset/arc-agi_training_challenges.json", "r") as fh:
            challenges = json.load(fh)
        with open("dataset/arc-agi_training_solutions.json", "r") as fh:
            solutions = json.load(fh)
        return challenges, solutions
    except FileNotFoundError:
        return {}, {}


def available_models(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    models = sorted({col[4:] for col in df.columns if col.startswith("acc_")})
    acc_cols = [f"acc_{m}" for m in models if f"acc_{m}" in df.columns]
    return models, acc_cols


@st.cache_data(show_spinner=False)
def apply_sql(df: pd.DataFrame, sql_text: str) -> pd.DataFrame:
    sql = sql_text.strip()
    if not sql:
        sql = "SELECT * FROM t"
    elif sql.lower().startswith("where"):
        sql = f"SELECT * FROM t {sql}"
    elif not sql.lower().startswith("select"):
        sql = f"SELECT * FROM t WHERE {sql}"
    con = duckdb.connect()
    con.register("t", df)
    out = con.execute(sql).fetch_df()
    con.close()
    return out


def lookup_row(df: pd.DataFrame, problem_id: str) -> Dict[str, Any]:
    hits = df.loc[df["problem_id"] == problem_id]
    return hits.iloc[0].to_dict() if not hits.empty else {}


def show_problem_overview(container, pid: str, challenge: Dict[str, Any], solution: List[list]) -> None:
    container.subheader(f"Problem {pid}")

    tests = challenge.get("test", []) if challenge else []
    for idx, test in enumerate(tests, start=1):
        container.write(f"**Test {idx}:**")
        cols = container.columns(2)
        img_in = grid_to_image(test.get("input"))
        if img_in is not None:
            cols[0].image(img_in, caption="Input", width="content")
        expected = None
        if solution and idx - 1 < len(solution):
            expected = solution[idx - 1]
        elif "output" in test:
            expected = test["output"]
        img_out = grid_to_image(expected)
        if img_out is not None:
            cols[1].image(img_out, caption="Expected", width="content")

    train_examples = (challenge or {}).get("train", [])[:3]
    for idx, example in enumerate(train_examples, start=1):
        container.write(f"**Train {idx}:**")
        cols = container.columns(2)
        img_in = grid_to_image(example.get("input"))
        if img_in is not None:
            cols[0].image(img_in, caption="Input", width="content")
        img_out = grid_to_image(example.get("output"))
        if img_out is not None:
            cols[1].image(img_out, caption="Output", width="content")


def show_model_attempts(
    row: Dict[str, Any], models: Iterable[str], expected_outputs: List[List[int]] | List[Any]
) -> None:
    assets = row.get("assets", {})
    solutions = assets.get("solutions", {})
    if len(solutions) == 0:
        st.info("No attempt grids found for this problem.")
        return

    models_with_data = [m for m in models if solutions.get(m)]
    if not models_with_data:
        st.info("No attempt grids found for this problem.")
        return

    tabs = st.tabs(models_with_data)
    for tab, model in zip(tabs, models_with_data):
        with tab:
            acc = float(row.get(f"acc_{model}", 0.0) or 0.0)
            corr = bool(row.get(f"correct_{model}", False))
            tab.write(f"**{model}** — Correct: {'✅' if corr else '❌'} — Accuracy: {int(round(acc * 100))}%")

            seeds = solutions.get(model, {})
            seed_options = sorted(seeds.keys(), key=lambda s: int(s))
            if not seed_options:
                tab.info("No seeds with attempts for this model.")
                continue

            selected_seed = tab.selectbox("Seed", seed_options, key=f"seed_{model}_{row['problem_id']}")
            seed_payload = seeds.get(selected_seed, {})
            attempts = seed_payload.get("attempts", [])
            if len(attempts) == 0:
                tab.info("No attempts recorded for this seed.")
                continue

            num_tests = len(attempts[0]) if len(attempts) > 0 and isinstance(attempts[0], (list, np.ndarray)) else 0
            for test_idx in range(num_tests):
                tab.write(f"**Test {test_idx + 1}:**")
                cols = tab.columns(len(attempts))
                expected_grid = expected_outputs[test_idx] if test_idx < len(expected_outputs) else None
                for attempt_idx, attempt in enumerate(attempts):
                    grid = attempt[test_idx] if test_idx < len(attempt) else None
                    # Convert numpy object arrays to proper lists for visualization
                    if grid is not None and isinstance(grid, np.ndarray) and grid.dtype == object:
                        grid = [list(row) if isinstance(row, np.ndarray) else row for row in grid]
                    img = grid_to_image(grid)
                    if img is not None:
                        if expected_grid is None:
                            status = None
                        else:
                            status = grids_equal(expected_grid, grid)
                        if status is True:
                            suffix = " ✅"
                        elif status is False:
                            suffix = " ❌"
                        else:
                            suffix = ""
                        cols[attempt_idx].image(
                            img,
                            caption=f"Attempt {attempt_idx + 1}{suffix}",
                            width="content",
                        )


TABLE_SESSION_KEY = "_arc_results_active_table"


def main() -> None:
    st.sidebar.subheader("Settings")
    parquet_path = st.sidebar.text_input("Parquet path (optional)", value="", key="parquet_path")

    st.sidebar.subheader("SQL Filter")
    st.sidebar.caption("Enter a full SELECT … or a WHERE fragment.")
    sql_in = st.sidebar.text_area("SQL (SELECT … or WHERE …)", value="", height=110, key="sql_input")

    table_height = st.sidebar.slider("Table height", min_value=320, max_value=900, value=520, step=20)

    st.sidebar.subheader("Export")
    default_csv_name = "arc_results_filtered.csv"
    csv_filename = st.sidebar.text_input(
        "CSV filename", value=default_csv_name, key="csv_filename"
    ).strip()
    save_csv_clicked = st.sidebar.button("Save table to CSV", key="save_csv")

    df = load_df(parquet_path or None)
    if df.empty:
        st.stop()

    # Load meta columns from the meta CSV
    meta_df = pd.read_csv("dataset_meta/arc_training_meta.csv")
    meta_cols = [c for c in meta_df.columns if c != "uid"]

    models, acc_cols = available_models(df)

    # Exclude assets column from table display
    scalar_cols = [c for c in df.columns if c != "assets"]

    try:
        df_filtered = apply_sql(df[scalar_cols], sql_in)
        if sql_in.strip():
            st.code(sql_in, language="sql")
    except Exception as exc:
        st.error(f"SQL error: {exc}")
        df_filtered = df[scalar_cols]

    # Reorder columns to match display order: problem_id, meta_cols, then rest
    display_order = []
    if "problem_id" in df_filtered.columns:
        display_order.append("problem_id")
    for meta_col in meta_cols:
        if meta_col in df_filtered.columns:
            display_order.append(meta_col)
    for col in df_filtered.columns:
        if col not in display_order:
            display_order.append(col)
    df_filtered = df_filtered[display_order]

    # Column visibility controls
    st.sidebar.subheader("Column Visibility")
    visible_cols = [col for col in df_filtered.columns if st.sidebar.checkbox(col, value=True, key=f"col_vis_{col}")]
    df_filtered = df_filtered[visible_cols]

    grid_builder = GridOptionsBuilder.from_dataframe(df_filtered)
    grid_builder.configure_default_column(filter=True, sortable=True, resizable=True, floatingFilter=True)
    grid_builder.configure_selection(selection_mode="single", use_checkbox=True)
    grid_builder.configure_grid_options(rowSelection="single")

    # Pin problem_id and meta columns to the left
    if "problem_id" in df_filtered.columns:
        grid_builder.configure_column("problem_id", pinned="left", width=130)
    for meta_col in meta_cols:
        if meta_col in df_filtered.columns:
            grid_builder.configure_column(meta_col, pinned="left")

    cols = st.columns([4, 1])
    cols[0].caption("Select a problem to view ARC grids and model attempts.")
    count_placeholder = cols[1].empty()

    resp = AgGrid(
        df_filtered,
        gridOptions=grid_builder.build(),
        height=table_height,
        theme="balham",
        allow_unsafe_jscode=False,
        update_on=["selectionChanged", "filterChanged", "sortChanged"],
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        key="arc_results_table",
    )

    # Capture the current grid view; fall back to previous view if no grid event triggered this run.
    grid_response = getattr(resp, "grid_response", {}) if resp is not None else {}
    if isinstance(grid_response, dict) and grid_response:
        data_obj = resp.data
        grid_df = data_obj.copy() if isinstance(data_obj, pd.DataFrame) else pd.DataFrame(data_obj)
        st.session_state[TABLE_SESSION_KEY] = grid_df
    else:
        cached_df = st.session_state.get(TABLE_SESSION_KEY)
        grid_df = cached_df.copy() if isinstance(cached_df, pd.DataFrame) else df_filtered.copy()
    grid_df = grid_df.reindex(columns=df_filtered.columns)

    count_placeholder.markdown(f"<div style='text-align: right; font-size: 0.8em;'>{len(grid_df)} rows × {len(grid_df.columns)} cols</div>", unsafe_allow_html=True)

    if save_csv_clicked:
        if not csv_filename:
            st.sidebar.error("Please provide a filename before saving.")
        else:
            csv_path = Path(csv_filename).expanduser()
            try:
                grid_df.to_csv(csv_path, index=False)
            except Exception as exc:
                st.sidebar.error(f"Failed to save CSV: {exc}")
            else:
                st.sidebar.success(f"Saved {len(grid_df)} rows to {csv_path}")

    sel = resp.get("selected_rows", [])
    if sel is None:
        sel = []
    elif isinstance(sel, pd.DataFrame):
        sel = sel.to_dict("records")
    elif isinstance(sel, dict):
        sel = [sel]

    viz_panel, main_panel = st.columns([1, 2], gap="medium")
    challenges, solutions = load_arc_dataset()

    if len(sel) == 1:
        pid = str(sel[0].get("problem_id", ""))
        row = lookup_row(df, pid)
        if row:
            challenge_obj = challenges.get(pid, {})
            solution_outputs = solutions.get(pid, [])
            show_problem_overview(viz_panel, pid, challenge_obj, solution_outputs)
            with main_panel:
                st.subheader("Model attempts")
                if "best_acc" in row:
                    st.write(f"Best Accuracy: **{round(float(row['best_acc']) * 100)}%**")
                if "any_correct" in row:
                    st.write("Any Correct: **" + ("✅" if row["any_correct"] else "❌") + "**")
                expected_outputs: List[Any] = []
                tests = challenge_obj.get("test", []) if isinstance(challenge_obj, dict) else []
                for idx, test in enumerate(tests):
                    expected = None
                    if solution_outputs and idx < len(solution_outputs):
                        expected = solution_outputs[idx]
                    elif isinstance(test, dict) and "output" in test:
                        expected = test["output"]
                    expected_outputs.append(expected)
                show_model_attempts(row, models, expected_outputs)
        else:
            viz_panel.info("Problem not found in dataframe.")


if __name__ == "__main__":
    main()
