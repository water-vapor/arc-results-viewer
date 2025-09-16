# app3.py — ARC Results Viewer with real data and on-demand rendering

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

import duckdb
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

from utils import grid_to_image, grids_equal


st.set_page_config(page_title="ARC Results Viewer", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def load_df(parquet_path: str | None) -> pd.DataFrame:
    """Load the processed parquet file and normalise JSON columns."""

    path = parquet_path or "arc_results_processed.parquet"
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        st.warning("No data found. Run preprocess_data.py first.")
        return pd.DataFrame()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return pd.DataFrame()

    def _as_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return {}
        return value if isinstance(value, dict) else {}

    if "assets" in df.columns:
        df["assets"] = df["assets"].map(_as_dict)
    else:
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


def available_models(df: pd.DataFrame) -> tuple[List[str], List[str], List[str]]:
    models = sorted({col[4:] for col in df.columns if col.startswith("acc_")})
    acc_cols = [f"acc_{m}" for m in models if f"acc_{m}" in df.columns]
    corr_cols = [f"correct_{m}" for m in models if f"correct_{m}" in df.columns]
    return models, acc_cols, corr_cols


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
    assets = row.get("assets") or {}
    solutions = assets.get("solutions") or {}
    if not solutions:
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

            seeds = solutions.get(model) or {}
            seed_options = sorted(seeds.keys(), key=lambda s: int(s))
            if not seed_options:
                tab.info("No seeds with attempts for this model.")
                continue

            selected_seed = tab.selectbox("Seed", seed_options, key=f"seed_{model}_{row['problem_id']}")
            seed_payload = seeds.get(selected_seed, {})
            attempts = seed_payload.get("attempts") or []
            if not attempts:
                tab.info("No attempts recorded for this seed.")
                continue

            num_tests = len(attempts[0]) if attempts and isinstance(attempts[0], list) else 0
            for test_idx in range(num_tests):
                tab.write(f"**Test {test_idx + 1}:**")
                cols = tab.columns(len(attempts))
                expected_grid = expected_outputs[test_idx] if test_idx < len(expected_outputs) else None
                for attempt_idx, attempt in enumerate(attempts):
                    grid = attempt[test_idx] if test_idx < len(attempt) else None
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


def main() -> None:
    st.sidebar.subheader("⚙️ Settings")
    parquet_path = st.sidebar.text_input("Parquet path (optional)", value="", key="parquet_path")

    st.sidebar.subheader("🔍 SQL Filter")
    st.sidebar.caption("Enter a full SELECT … or a WHERE fragment.")
    sql_in = st.sidebar.text_area("SQL (SELECT … or WHERE …)", value="", height=110, key="sql_input")

    table_height = st.sidebar.slider("Table height", min_value=320, max_value=900, value=520, step=20)

    df = load_df(parquet_path or None)
    if df.empty:
        st.stop()

    models, acc_cols, corr_cols = available_models(df)
    scalar_cols = [
        c
        for c in ["problem_id", "title", "category", "best_acc", "any_correct", *acc_cols, *corr_cols]
        if c in df.columns
    ]

    try:
        df_filtered = apply_sql(df[scalar_cols], sql_in)
        if sql_in.strip():
            st.code(sql_in, language="sql")
    except Exception as exc:
        st.error(f"SQL error: {exc}")
        df_filtered = df[scalar_cols]

    grid_builder = GridOptionsBuilder.from_dataframe(df_filtered)
    grid_builder.configure_default_column(filter=True, sortable=True, resizable=True, floatingFilter=True)
    grid_builder.configure_selection(selection_mode="single", use_checkbox=True)
    grid_builder.configure_grid_options(rowSelection="single")
    for col, opts in {
        "problem_id": {"pinned": "left", "width": 130},
        "title": {"pinned": "left", "width": 220},
        "category": {"width": 130},
    }.items():
        if col in df_filtered.columns:
            grid_builder.configure_column(col, **opts)

    st.caption("Select a problem to view ARC grids and model attempts.")
    resp = AgGrid(
        df_filtered,
        gridOptions=grid_builder.build(),
        height=table_height,
        theme="balham",
        allow_unsafe_jscode=False,
        update_on=["selectionChanged"],
    )

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
