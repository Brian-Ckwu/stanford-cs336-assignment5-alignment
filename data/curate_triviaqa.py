"""Curate all unique TriviaQA training questions and a validation sample.

Run from the repository root: uv run python data/curate_triviaqa.py
"""

import argparse
import json
import random
from pathlib import Path


def deduplicate(rows):
    seen = set()
    result = []
    for row in rows:
        question_id = row["question_id"]
        if question_id in seen:
            continue
        seen.add(question_id)
        answer = row["answer"]
        result.append({
            "dataset": "triviaqa",
            "question_id": question_id,
            "question": row["question"],
            "answer": {
                "value": answer["value"],
                "aliases": answer.get("aliases", []),
                "normalized_aliases": answer.get("normalized_aliases", []),
                "normalized_value": answer.get("normalized_value", ""),
            },
        })
    return result


def curate(train, validation, output_dir: Path, seed: int = 42):
    train_rows = deduplicate(train)
    validation_rows = deduplicate(validation)
    if {row["question_id"] for row in train_rows} & {row["question_id"] for row in validation_rows}:
        raise ValueError("Training and validation question IDs overlap")
    test_rows = random.Random(seed).sample(validation_rows, 1000)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("test", test_rows)):
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Train: {len(train)} source rows -> {len(train_rows)} unique questions")
    print(f"Validation: {len(validation)} source rows -> {len(validation_rows)} unique questions -> {len(test_rows)} sampled (seed={seed})")


def main():
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "triviaqa")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    curate(dataset["train"], dataset["validation"], args.output_dir, args.seed)


if __name__ == "__main__":
    main()
