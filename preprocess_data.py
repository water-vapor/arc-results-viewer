import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm


MAX_SEEDS_PER_MODEL = 10  # keep a handful of samples per model for the UI


def load_model_results() -> Dict[str, Dict[str, List[list]]]:
    """Return `{model_name: {problem_id: predictions}}` for every JSON file."""
    model_results: Dict[str, Dict[str, List[list]]] = {}
    for json_file in Path("results_json").glob("*.json"):
        with open(json_file, "r") as fh:
            model_results[json_file.stem] = json.load(fh)
    return model_results


def summarise_predictions(predictions: Iterable[list]) -> Tuple[float, bool]:
    """Return (mean accuracy, any_correct) for a list of predictions."""
    preds = list(predictions)
    if not preds:
        return 0.0, False
    flags = [bool(p[1]) for p in preds]
    return sum(flags) / len(preds), any(flags)


def collect_seed_attempts(predictions: Iterable[list]) -> Dict[str, Dict[str, object]]:
    """Return `{seed: {"correct": bool, "attempts": [...]}}` for the UI."""
    seeds: Dict[str, Dict[str, object]] = {}
    for pred in list(predictions)[:MAX_SEEDS_PER_MODEL]:
        seed_id = str(pred[0])
        seeds[seed_id] = {
            "correct": bool(pred[1]),
            "attempts": pred[2] if len(pred) > 2 and pred[2] else [],
        }
    return {k: v for k, v in seeds.items() if v["attempts"]}


def process_problem(
    problem_id: str,
    model_results: Dict[str, Dict[str, list]],
    model_names: Iterable[str],
) -> Dict[str, object]:
    """Collect flattened metrics + raw attempts for one ARC problem."""
    row: Dict[str, object] = {
        "problem_id": problem_id,
        "title": f"ARC Problem {problem_id}",
        "category": "training",
    }

    best_acc = 0.0
    any_correct_global = False
    solutions_payload: Dict[str, Dict[str, object]] = {}

    for model in model_names:
        preds = (model_results.get(model) or {}).get(problem_id, [])
        acc, any_correct = summarise_predictions(preds)
        row[f"acc_{model}"] = acc
        row[f"correct_{model}"] = any_correct

        best_acc = max(best_acc, acc)
        any_correct_global = any_correct_global or any_correct

        seed_attempts = collect_seed_attempts(preds)
        if seed_attempts:
            solutions_payload[model] = seed_attempts

    row["best_acc"] = best_acc
    row["any_correct"] = any_correct_global
    row["assets"] = json.dumps({"solutions": solutions_payload})
    return row


def main() -> None:
    print("Loading model outputs...")
    model_results = load_model_results()
    model_names = sorted(model_results.keys())

    with open("dataset/arc-agi_training_challenges.json", "r") as fh:
        challenge_ids = sorted(json.load(fh).keys())

    print(f"Processing {len(challenge_ids)} problems across {len(model_names)} models...")

    rows = []
    for problem_id in tqdm(challenge_ids, desc="Problems", unit="problem"):
        rows.append(process_problem(problem_id, model_results, model_names))

    df = pd.DataFrame(rows)
    df.to_parquet("arc_results_processed.parquet", index=False)

    print(f"\nSaved {len(df)} problems to arc_results_processed.parquet")
    print(f"Models: {model_names}")
    print(f"Avg accuracy: {df['best_acc'].mean():.1%}")
    print(f"Solved problems: {int(df['any_correct'].sum())}/{len(df)}")


if __name__ == "__main__":
    main()
