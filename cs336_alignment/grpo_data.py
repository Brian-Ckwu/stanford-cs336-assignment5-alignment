"""CPU-only dataset loading and prompt preparation for GRPO."""

import json
from pathlib import Path


def load_rows(dataset_folder: str | Path) -> tuple[list[dict], str]:
    path = Path(dataset_folder) / "train.jsonl"
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Empty training dataset: {path}")
    kinds = {row.get("dataset", "gsm8k") for row in rows}
    if len(kinds) != 1 or not kinds <= {"gsm8k", "triviaqa"}:
        raise ValueError(f"Unsupported or mixed dataset identifiers in {path}: {kinds}")
    return rows, kinds.pop()


def resolve_prompt_path(prompt_path: str | None, dataset: str) -> Path:
    if prompt_path is not None:
        return Path(prompt_path)
    name = "qwen3_recommended_prompt_math_with_cc.prompt" if dataset == "triviaqa" else "r1_zero.prompt"
    return Path(__file__).resolve().parent / "prompts" / name


def prepare_rows(rows: list[dict], dataset: str, template: str, group_size: int) -> list[dict]:
    prepared = []
    for row in rows:
        row = dict(row)
        row["prompt"] = template.replace("{question}", row["question"]).replace("{group_size}", str(group_size))
        if dataset == "gsm8k":
            row["answer"] = row["answer"].split("####")[-1].strip()
        prepared.append(row)
    return prepared
