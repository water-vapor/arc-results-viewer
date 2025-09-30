import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

from utils import normalize_grid


MAX_SEEDS_PER_MODEL = 100  # keep a handful of samples per model for the UI


def load_model_results() -> Dict[str, Dict[str, List[list]]]:
    return {f.stem: json.load(open(f)) for f in Path("results_json").glob("*.json")}


def load_ground_truth() -> Dict[str, List[list]]:
    return json.load(open("dataset/arc-agi_training_solutions.json"))


def check_correctness(prediction: list, ground_truth: List[list]) -> bool:
    if len(prediction) <= 2 or not prediction[2] or len(prediction[2]) < 2:
        return False
    gt_grids = [normalize_grid(g) for g in ground_truth]
    attempts = [[normalize_grid(g) for g in att] for att in prediction[2]]
    for i, target in enumerate(gt_grids):
        candidates = [att[i] for att in attempts if i < len(att)]
        if not candidates or all(c != target for c in candidates):
            return False
    return True


def summarise_predictions(
    predictions: Iterable[list], problem_id: str, ground_truth: Dict[str, List[list]]
) -> Tuple[float, bool]:
    preds = list(predictions)
    if not preds:
        return 0.0, False
    flags = [check_correctness(p, ground_truth[problem_id]) for p in preds] if problem_id in ground_truth else [bool(p[1]) for p in preds]
    return sum(flags) / len(preds), any(flags)


def collect_seed_attempts(
    predictions: Iterable[list], problem_id: str, ground_truth: Dict[str, List[list]]
) -> Dict[str, Dict[str, object]]:
    seeds = {}
    for pred in list(predictions)[:MAX_SEEDS_PER_MODEL]:
        correct = check_correctness(pred, ground_truth[problem_id]) if problem_id in ground_truth else bool(pred[1])
        attempts = pred[2] if len(pred) > 2 and pred[2] else []
        if attempts:
            seeds[str(pred[0])] = {"correct": correct, "attempts": attempts}
    return seeds


def process_problem(
    problem_id: str,
    model_results: Dict[str, Dict[str, list]],
    model_names: Iterable[str],
    ground_truth: Dict[str, List[list]],
) -> Dict[str, object]:
    row = {"problem_id": problem_id}
    best_acc = 0.0
    any_correct_global = False
    solutions_payload = {}

    for model in model_names:
        preds = model_results.get(model, {}).get(problem_id, [])
        acc, any_correct = summarise_predictions(preds, problem_id, ground_truth)
        row[f"acc_{model}"] = acc
        row[f"correct_{model}"] = any_correct
        best_acc = max(best_acc, acc)
        any_correct_global = any_correct_global or any_correct

        seed_attempts = collect_seed_attempts(preds, problem_id, ground_truth)
        if seed_attempts:
            solutions_payload[model] = seed_attempts

    row["best_acc"] = best_acc
    row["any_correct"] = any_correct_global
    row["assets"] = {"solutions": solutions_payload}
    return row


def main() -> None:
    model_results = load_model_results()
    model_names = sorted(model_results.keys())
    ground_truth = load_ground_truth()
    challenge_ids = sorted(json.load(open("dataset/arc-agi_training_challenges.json")).keys())

    rows = [process_problem(pid, model_results, model_names, ground_truth)
            for pid in tqdm(challenge_ids, desc="Problems")]

    df = pd.DataFrame(rows)
    df["problem_id"] = df["problem_id"].astype(str)

    meta_df = pd.read_csv("dataset_meta/arc_training_meta.csv")
    meta_df = meta_df.rename(columns={"uid": "problem_id"})
    meta_df["problem_id"] = meta_df["problem_id"].astype(str)
    df = df.merge(meta_df, on="problem_id", how="left")

    df.to_parquet("arc_results_processed.parquet", index=False)

    for model in model_names:
        acc = df[f"acc_{model}"].mean()
        correct = df[f"correct_{model}"].mean()
        print(f"{model}: accuracy {acc:.1%}, any correct {correct:.1%}")

    acc_cols = [f"acc_{model}" for model in model_names]
    correct_cols = [f"correct_{model}" for model in model_names]
    print(f"\nOverall: accuracy {df[acc_cols].to_numpy().mean():.1%}, any correct {df[correct_cols].any(axis=1).mean():.1%}")


if __name__ == "__main__":
    main()
