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


def require_nonempty_string(value: Any, *, description: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string, got {type(value).__name__}")
    value = value.strip()
    if not value:
        raise ValueError(f"{description} must not be empty")
    return value


def extract_user_content(prompt: Any, *, example_id: str) -> str:
    """Return the single user message without assuming it is prompt[0]."""
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(
            f"prompt must be a nonempty message list for example_id={example_id!r}"
        )

    user_messages = [
        message
        for message in prompt
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(user_messages) != 1:
        raise ValueError(
            "Expected exactly one user message for "
            f"example_id={example_id!r}, found {len(user_messages)}"
        )

    return require_nonempty_string(
        user_messages[0].get("content"),
        description=f"user content for example_id={example_id!r}",
    )


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    ground_truth: dict[str, dict[str, Any]] = {}
    for data in read_jsonl(path):
        example_id = data["example_id"]
        if example_id in ground_truth:
            raise ValueError(f"Duplicate example_id={example_id!r} in {path}")

        ground_truth[example_id] = {
            "question": require_nonempty_string(
                data.get("question"),
                description=f"question for example_id={example_id!r}",
            ),
            "num_samples": data["num_samples"],
            "num_correct": data["num_correct"],
        }
    return ground_truth


def validate_question_in_user_content(
    question: str,
    user_content: str,
    *,
    example_id: str,
) -> None:
    """Ensure the rendered user message contains the authoritative question intact."""
    if user_content == question:
        return
    if user_content.startswith(question + "\n\n"):
        return
    raise ValueError(
        "Ground-truth question does not match the user prompt for "
        f"example_id={example_id!r}"
    )


def build_rows(input_folder: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    sampled_path = input_folder / "sampled.jsonl"
    ground_truth_path = input_folder / "ground_truth.jsonl"
    ground_truth = load_ground_truth(ground_truth_path)

    for data in read_jsonl(sampled_path):
        example_id = data["example_id"]
        if example_id not in ground_truth:
            raise ValueError(
                f"example_id={example_id!r} is in sampled.jsonl but not ground_truth.jsonl"
            )

        question = ground_truth[example_id]["question"]
        user_content = extract_user_content(data["prompt"], example_id=example_id)
        validate_question_in_user_content(question, user_content, example_id=example_id)
        correctness = data["correctness"]

        if example_id in rows:
            if user_content != rows[example_id]["_user_content"]:
                raise ValueError(
                    f"Mismatched user content for example_id={example_id!r}"
                )
        else:
            rows[example_id] = {
                "num_samples": 0,
                "num_correct": 0,
                "query": question,
                "_user_content": user_content,
            }

        rows[example_id]["num_samples"] += 1
        rows[example_id]["num_correct"] += correctness

    for example_id, expected in ground_truth.items():
        if example_id not in rows:
            raise ValueError(
                f"example_id={example_id!r} is in ground_truth.jsonl but not sampled.jsonl"
            )

        row = rows[example_id]
        if (
            row["num_samples"] != expected["num_samples"]
            or row["num_correct"] != expected["num_correct"]
        ):
            raise ValueError(
                "Count mismatch for "
                f"example_id={example_id!r}: sampled has "
                f"num_samples={row['num_samples']}, num_correct={row['num_correct']}; "
                f"ground truth has num_samples={expected['num_samples']}, "
                f"num_correct={expected['num_correct']}"
            )

        row["expected_accuracy"] = row["num_correct"] / row["num_samples"]
        del row["_user_content"]

    unique_queries = {row["query"] for row in rows.values()}
    if len(rows) > 1 and len(unique_queries) == 1:
        raise ValueError(
            f"All {len(rows)} rows contain the same query; "
            "question extraction is probably incorrect"
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
    unique_queries = len({row["query"] for row in rows.values()})
    print(
        f"Wrote {len(rows)} rows ({unique_queries} unique queries) to {output_path}"
    )


if __name__ == "__main__":
    main()
