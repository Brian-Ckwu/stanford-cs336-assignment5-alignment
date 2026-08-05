#!/usr/bin/env python3
"""Standardize capability calibration data for GRPO training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert sampled.jsonl and ground_truth.jsonl in an experiment folder "
            "to the GRPO dataset JSONL format used by the calibration notebook."
        )
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        help="Folder containing sampled.jsonl and ground_truth.jsonl.",
    )
    parser.add_argument(
        "--output_filename",
        default="grpo_dataset.jsonl",
        help="Output JSONL filename to write inside input_folder.",
    )
    return parser.parse_args()


def resolve_input_folder(input_folder: str) -> Path:
    path = Path(input_folder).expanduser()
    if path.exists():
        return path
    if path.is_absolute():
        raise FileNotFoundError(f"Input folder does not exist: {path}")

    # Let the notebook's relative paths work from the script directory, while
    # also supporting invocation from the project root or nearby subdirectories.
    search_roots = [Path.cwd(), Path(__file__).resolve().parent]
    for root in list(search_roots):
        search_roots.extend(root.parents)

    seen: set[Path] = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = root / path
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Input folder does not exist: {path}")


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def build_rows(input_folder: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    sampled_path = input_folder / "sampled.jsonl"
    ground_truth_path = input_folder / "ground_truth.jsonl"

    for data in read_jsonl(sampled_path):
        example_id = data["example_id"]
        query = data["prompt"][0]["content"].split("\n\n")[0]
        correctness = data["correctness"]

        if example_id in rows:
            if query != rows[example_id]["query"]:
                raise ValueError(f"Mismatched query for example_id={example_id!r}")
        else:
            rows[example_id] = {"num_samples": 0, "num_correct": 0}
            rows[example_id]["query"] = query

        rows[example_id]["num_samples"] += 1
        rows[example_id]["num_correct"] += correctness

    for data in read_jsonl(ground_truth_path):
        example_id = data["example_id"]
        if example_id not in rows:
            raise ValueError(f"example_id={example_id!r} is in ground_truth.jsonl but not sampled.jsonl")

        row = rows[example_id]
        if row["num_samples"] != data["num_samples"] or row["num_correct"] != data["num_correct"]:
            raise ValueError(
                "Count mismatch for "
                f"example_id={example_id!r}: sampled has "
                f"num_samples={row['num_samples']}, num_correct={row['num_correct']}; "
                f"ground truth has num_samples={data['num_samples']}, "
                f"num_correct={data['num_correct']}"
            )

        row["expected_accuracy"] = row["num_correct"] / row["num_samples"]

    missing_accuracy = [example_id for example_id, row in rows.items() if "expected_accuracy" not in row]
    if missing_accuracy:
        raise ValueError(
            "Some sampled examples are missing from ground_truth.jsonl: "
            + ", ".join(missing_accuracy[:10])
        )

    return rows


def write_rows(rows: dict[str, dict[str, Any]], output_path: Path) -> None:
    with output_path.open(mode="wt") as f:
        for example_id, values in rows.items():
            f.write(json.dumps({"example_id": example_id, **values}) + "\n")


def main() -> None:
    args = parse_args()
    input_folder = resolve_input_folder(args.input_folder)
    output_path = input_folder / args.output_filename
    rows = build_rows(input_folder)
    write_rows(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
