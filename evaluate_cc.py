#!/usr/bin/env python3
"""Evaluate capability-calibration confidence predictions with vLLM."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


DEFAULT_METRICS_SRC = "/nfs/brian-wu/llm-calibration/src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate confidence scores for capability-calibration examples and "
            "evaluate them against expected_accuracy."
        )
    )
    parser.add_argument("--model_folder", required=True, help="HF/vLLM model folder to evaluate.")
    parser.add_argument("--test_data_jsonl", required=True, help="JSONL with query and expected_accuracy fields.")
    parser.add_argument("--prompt_path", required=True, help="Prompt template path containing {question}.")
    parser.add_argument("--vllm_device", required=True, help="CUDA device id, or comma-separated ids, for vLLM.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--metrics_src_path", default=DEFAULT_METRICS_SRC)
    parser.add_argument("--output_jsonl", default=None, help="Optional path for per-example generations.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most this many rows.")
    parser.add_argument("--start_index", type=int, default=0, help="First row index after optional shuffle.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle rows with --seed before applying --start_index/--limit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--stop", default="</answer>")
    parser.add_argument(
        "--include_stop_str_in_output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--invalid_policy",
        choices=("zero", "nan", "error"),
        default="zero",
        help="How to handle outputs with no parseable score.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--vllm_logging_level", default="ERROR")
    return parser.parse_args()


def resolve_existing_path(path: str, *, repo_root: Path) -> Path:
    candidates = [Path(path)]
    if not Path(path).is_absolute():
        candidates.extend(
            [
                repo_root / path,
                repo_root / "cs336_alignment" / path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find path: {path}")


def load_prompt_template(prompt_path: Path) -> str:
    template = prompt_path.read_text()
    if "{question}" not in template:
        raise ValueError(f"Prompt template must contain '{{question}}': {prompt_path}")
    return template


def load_dataset(path: Path, *, seed: int, shuffle: bool, start_index: int, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "query" not in row:
                raise KeyError(f"Missing 'query' in {path}:{line_number}")
            if "expected_accuracy" not in row:
                raise KeyError(f"Missing 'expected_accuracy' in {path}:{line_number}")
            row["_source_line"] = line_number
            row["expected_accuracy"] = float(row["expected_accuracy"])
            if not 0.0 <= row["expected_accuracy"] <= 1.0:
                raise ValueError(f"expected_accuracy outside [0, 1] in {path}:{line_number}")
            rows.append(row)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    if start_index < 0:
        raise ValueError("--start_index must be non-negative.")
    rows = rows[start_index:]
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative.")
        rows = rows[:limit]
    return rows


def build_prompts(rows: list[dict[str, Any]], prompt_template: str) -> list[str]:
    return [prompt_template.format(question=row["query"]) for row in rows]


def extract_boxed_answer(text: str) -> str | None:
    start = text.find(r"\boxed{")
    if start < 0:
        return None
    i = start + len(r"\boxed{")
    depth = 1
    chars = []
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(char)
        else:
            chars.append(char)
        i += 1
    return None


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def parse_confidence(response: str) -> tuple[float | None, str | None]:
    """Extract a confidence in [0, 1] from the model response."""
    text = response.strip()
    answer_text: str | None = None

    answer_match = re.search(r"<answer>\s*(.*?)\s*(?:</answer>|$)", text, flags=re.DOTALL)
    if answer_match is not None:
        answer_text = answer_match.group(1).strip()
    else:
        boxed = extract_boxed_answer(text)
        if boxed is not None:
            answer_text = boxed

    if answer_text is None:
        return None, "missing_answer_tag"

    boxed = extract_boxed_answer(answer_text)
    if boxed is not None:
        answer_text = boxed

    try:
        value = float(answer_text)
    except ValueError:
        numbers = _NUMBER_RE.findall(answer_text)
        if len(numbers) != 1:
            return None, "answer_not_single_number"
        value = float(numbers[0])

    if not math.isfinite(value):
        return None, "answer_not_finite"
    if value < 0.0 or value > 1.0:
        return None, "answer_out_of_range"
    return value, None


def add_metrics_to_path(metrics_src_path: Path) -> None:
    if not metrics_src_path.exists():
        raise FileNotFoundError(f"Metrics source path does not exist: {metrics_src_path}")
    sys.path.insert(0, str(metrics_src_path))


def compute_metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    from metrics.custom import c_star_metrics

    metrics = c_star_metrics(predictions, targets)
    return asdict(metrics)


def infer_tensor_parallel_size(vllm_device: str, tensor_parallel_size: int | None) -> int:
    if tensor_parallel_size is not None:
        return tensor_parallel_size
    devices = [device.strip() for device in vllm_device.split(",") if device.strip()]
    return max(1, len(devices))


def generate_with_vllm(args: argparse.Namespace, prompts: list[str]) -> list[str]:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.vllm_device
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ["VLLM_LOGGING_LEVEL"] = args.vllm_logging_level

    from vllm import LLM, SamplingParams

    sampling_kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    if args.stop:
        sampling_kwargs["stop"] = args.stop
        sampling_kwargs["include_stop_str_in_output"] = args.include_stop_str_in_output
    sampling_params = SamplingParams(**sampling_kwargs)

    llm_kwargs: dict[str, Any] = {
        "model": args.model_folder,
        "tensor_parallel_size": infer_tensor_parallel_size(args.vllm_device, args.tensor_parallel_size),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    for output in outputs:
        if len(output.outputs) != 1:
            raise RuntimeError(f"Expected one completion per prompt, got {len(output.outputs)}.")

    return [output.outputs[0].text for output in outputs]


def materialize_predictions(
    rows: list[dict[str, Any]],
    prompts: list[str],
    responses: list[str],
    invalid_policy: str,
) -> tuple[list[dict[str, Any]], list[float], list[float], int]:
    if not (len(rows) == len(prompts) == len(responses)):
        raise ValueError("Rows, prompts, and responses must have the same length.")

    records = []
    predictions = []
    targets = []
    invalid_count = 0
    for row, prompt, response in zip(rows, prompts, responses):
        parsed, parse_error = parse_confidence(response)
        if parsed is None:
            invalid_count += 1
            if invalid_policy == "error":
                raise ValueError(
                    "Could not parse confidence for "
                    f"{row.get('example_id', row.get('_source_line'))}: {parse_error}\n{response}"
                )
            if invalid_policy == "zero":
                parsed_for_metrics = 0.0
            elif invalid_policy == "nan":
                parsed_for_metrics = float("nan")
            else:
                raise ValueError(f"Unknown invalid_policy: {invalid_policy}")
        else:
            parsed_for_metrics = parsed

        record = {
            "example_id": row.get("example_id"),
            "source_line": row.get("_source_line"),
            "query": row["query"],
            "expected_accuracy": row["expected_accuracy"],
            "prompt": prompt,
            "response": response,
            "parsed_confidence": parsed,
            "prediction_for_metrics": parsed_for_metrics,
            "parse_error": parse_error,
        }
        records.append(record)
        predictions.append(parsed_for_metrics)
        targets.append(row["expected_accuracy"])

    if invalid_policy == "nan":
        filtered = [
            (prediction, target)
            for prediction, target in zip(predictions, targets)
            if math.isfinite(prediction)
        ]
        if not filtered:
            raise ValueError("No valid parsed predictions.")
        predictions, targets = [list(values) for values in zip(*filtered)]

    return records, predictions, targets, invalid_count


def write_jsonl(path: Path, records: list[dict[str, Any]], metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
        f.write(json.dumps({"metrics": metrics}) + "\n")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    args.model_folder = str(resolve_existing_path(args.model_folder, repo_root=repo_root))
    test_data_jsonl = resolve_existing_path(args.test_data_jsonl, repo_root=repo_root)
    prompt_path = resolve_existing_path(args.prompt_path, repo_root=repo_root)
    metrics_src_path = resolve_existing_path(args.metrics_src_path, repo_root=repo_root)

    prompt_template = load_prompt_template(prompt_path)
    rows = load_dataset(
        test_data_jsonl,
        seed=args.seed,
        shuffle=args.shuffle,
        start_index=args.start_index,
        limit=args.limit,
    )
    prompts = build_prompts(rows, prompt_template)
    print(f"Loaded {len(rows)} examples from {test_data_jsonl}")
    print(f"Using prompt template {prompt_path}")
    print(f"Using model {args.model_folder} on CUDA_VISIBLE_DEVICES={args.vllm_device}")

    responses = generate_with_vllm(args, prompts)
    records, predictions, targets, invalid_count = materialize_predictions(
        rows,
        prompts,
        responses,
        args.invalid_policy,
    )

    add_metrics_to_path(metrics_src_path)
    metrics = compute_metrics(predictions, targets)
    metrics["num_invalid"] = invalid_count
    metrics["num_scored"] = len(predictions)
    metrics["num_total"] = len(rows)

    if args.output_jsonl is not None:
        output_path = Path(args.output_jsonl).resolve()
        write_jsonl(output_path, records, metrics)
        print(f"Wrote generations to {output_path}")

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"MSE: {metrics['mse']:.8f}")
    print(f"Spearman: {metrics['spearman_r']:.8f}")


if __name__ == "__main__":
    main()
