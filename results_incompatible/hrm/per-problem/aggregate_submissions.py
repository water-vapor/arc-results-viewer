#!/usr/bin/env python3
"""Aggregate per-problem submissions from the latest evaluator step."""

from __future__ import annotations

import json
import re
from pathlib import Path


STEP_PATTERN = re.compile(r"evaluator_ARC_step_(\d+)$")


def collect_latest_submission(problem_dir: Path) -> tuple[str, dict]:
    """Return the problem hash and its latest submission payload."""
    step_dirs = []
    for child in problem_dir.iterdir():
        if not child.is_dir():
            continue
        match = STEP_PATTERN.match(child.name)
        if match:
            step_dirs.append((int(match.group(1)), child))
    if not step_dirs:
        raise RuntimeError(f"No evaluator step directories found in {problem_dir}")

    _, latest_dir = max(step_dirs, key=lambda item: item[0])
    submission_path = latest_dir / "submission.json"
    if not submission_path.is_file():
        raise FileNotFoundError(f"Missing submission.json at {submission_path}")

    with submission_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if len(data) != 1:
        raise ValueError(f"Expected a single key in {submission_path}, found {len(data)}")

    (problem_hash, submissions_list) = next(iter(data.items()))
    if problem_hash != problem_dir.name:
        raise ValueError(
            f"Problem hash mismatch: directory {problem_dir.name} vs key {problem_hash}"
        )
    if not isinstance(submissions_list, list):
        raise TypeError(f"Submission payload for {problem_hash} is not a list")
    if not submissions_list:
        raise ValueError(f"No submissions found for {problem_hash}")

    for idx, payload in enumerate(submissions_list):
        if not isinstance(payload, dict):
            raise TypeError(
                f"Submission entry {idx} for {problem_hash} is not a dict"
            )
        if len(payload) != 100:
            raise ValueError(
                f"Submission dict {idx} for {problem_hash} should contain 100 attempts,"
                f" found {len(payload)}"
            )

    return problem_hash, submissions_list


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    raw_data_dir = base_dir / "raw_data"
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"Missing raw_data directory at {raw_data_dir}")

    aggregate: dict[str, list] = {}
    for problem_dir in sorted(p for p in raw_data_dir.iterdir() if p.is_dir()):
        problem_hash, payload = collect_latest_submission(problem_dir)
        aggregate[problem_hash] = payload

    assert len(aggregate) == 400, f"Expected 400 problems, found {len(aggregate)}"

    output_path = base_dir / "submission.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {len(aggregate)} submissions to {output_path}")


if __name__ == "__main__":
    main()
