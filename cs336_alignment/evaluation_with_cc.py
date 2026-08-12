#!/usr/bin/env python3
"""Self-contained capability-calibration evaluation utilities and CLI."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

try:
    from .calibration_utils import (
        ConfidenceOutputFormat,
        configure_chat_template,
        parse_confidence,
        render_user_prompt,
    )
except ImportError:  # Support direct execution from cs336_alignment/.
    from calibration_utils import (
        ConfidenceOutputFormat,
        configure_chat_template,
        parse_confidence,
        render_user_prompt,
    )


DEFAULT_SAMPLING_PARAMS: dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 256,
    "n": 1,
    "seed": 42,
    "stop": "</answer>",
    "include_stop_str_in_output": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate capability-calibration predictions against expected_accuracy."
    )
    parser.add_argument("--model_folder", required=True, help="HF/vLLM model folder to evaluate.")
    parser.add_argument(
        "--test_data_jsonl",
        required=True,
        nargs="+",
        help="One or more JSONL files with query and expected_accuracy fields.",
    )
    parser.add_argument("--prompt_path", required=True, help="Prompt template containing {question}.")
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render each formatted prompt as one user chat message.",
    )
    parser.add_argument(
        "--chat_template_path",
        default=None,
        help="Optional Jinja chat-template override. Requires --use_chat_template.",
    )
    parser.add_argument(
        "--confidence_output_format",
        choices=("auto", "answer_tags", "boxed"),
        default="auto",
        help="Output contract used when parsing confidence predictions.",
    )
    parser.add_argument("--vllm_device", default=0, help="CUDA device id, or comma-separated ids.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_metrics_jsonl", default="cc_metrics.json")
    parser.add_argument("--cc_prediction_traces_jsonl", default="cc_prediction_traces.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
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
        candidates.extend([repo_root / path, repo_root / "cs336_alignment" / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find path: {path}")


def resolve_existing_paths(paths: Sequence[str], *, repo_root: Path) -> list[Path]:
    resolved = []
    missing = []
    for path in paths:
        try:
            resolved.append(resolve_existing_path(path, repo_root=repo_root))
        except FileNotFoundError:
            missing.append(path)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Could not find dataset path(s):\n{formatted}")
    return resolved


def validate_output_filename(filename: str, *, argument_name: str) -> str:
    path = Path(filename)
    if not filename or path.name != filename or filename in {".", ".."}:
        raise ValueError(f"{argument_name} must be a filename without a directory: {filename!r}")
    return filename


def build_output_paths(dataset_paths: Sequence[Path], filename: str) -> list[Path]:
    output_paths = [dataset_path.parent / filename for dataset_path in dataset_paths]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError(
            "Multiple input datasets would write to the same output path. "
            "Use at most one dataset from each parent directory."
        )
    return output_paths


def load_prompt_template(prompt_path: Path) -> str:
    template = prompt_path.read_text()
    if "{question}" not in template:
        raise ValueError(f"Prompt template must contain '{{question}}': {prompt_path}")
    return template


def _validate_target(value: Any, *, description: str) -> float:
    target = float(value)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError(f"{description} must be finite and in [0, 1], got {value!r}.")
    return target


def load_dataset(
    path: Path,
    *,
    seed: int,
    shuffle: bool,
    start_index: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows = []
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "query" not in row:
                raise KeyError(f"Missing 'query' in {path}:{line_number}")
            if "expected_accuracy" not in row:
                raise KeyError(f"Missing 'expected_accuracy' in {path}:{line_number}")
            row["_source_line"] = line_number
            row["expected_accuracy"] = _validate_target(
                row["expected_accuracy"],
                description=f"expected_accuracy in {path}:{line_number}",
            )
            rows.append(row)
    if shuffle:
        random.Random(seed).shuffle(rows)
    if start_index < 0:
        raise ValueError("--start_index must be non-negative.")
    rows = rows[start_index:]
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative.")
        rows = rows[:limit]
    return rows


def build_prompts(
    rows: Sequence[Mapping[str, Any]],
    prompt_template: str,
    *,
    tokenizer: Any | None = None,
    use_chat_template: bool = False,
) -> list[str]:
    return [
        render_user_prompt(
            tokenizer,
            prompt_template.format(question=row["query"]),
            use_chat_template=use_chat_template,
        )
        for row in rows
    ]


def _rank(values: np.ndarray) -> np.ndarray:
    sorted_indices = np.argsort(values)
    ranks = np.zeros_like(values, dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) - 1 and values[sorted_indices[end]] == values[sorted_indices[end + 1]]:
            end += 1
        ranks[sorted_indices[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt((x_centered**2).sum() * (y_centered**2).sum())
    if denominator == 0:
        return 0.0
    return float((x_centered * y_centered).sum() / denominator)


def _c_star_ece(predictions: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    if len(predictions) == 0:
        return 0.0
    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        upper_comparison = (
            predictions <= boundaries[index + 1]
            if index == n_bins - 1
            else predictions < boundaries[index + 1]
        )
        mask = (predictions >= boundaries[index]) & upper_comparison
        bin_size = int(mask.sum())
        if bin_size:
            ece += (bin_size / len(predictions)) * abs(targets[mask].mean() - predictions[mask].mean())
    return float(ece)


def compute_metrics(predictions: Sequence[float], targets: Sequence[float]) -> dict[str, float | int]:
    """Compute c(x) versus c*(x) metrics without an external metrics package."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    if len(predictions) == 0:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "pearson_r": 0.0,
            "spearman_r": 0.0,
            "ece": 0.0,
            "num_examples": 0,
        }
    prediction_array = np.asarray(predictions, dtype=float)
    target_array = np.asarray(targets, dtype=float)
    errors = prediction_array - target_array
    mse = float((errors**2).mean())
    return {
        "mae": float(np.abs(errors).mean()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "pearson_r": _pearson_correlation(prediction_array, target_array),
        "spearman_r": _pearson_correlation(_rank(prediction_array), _rank(target_array)),
        "ece": _c_star_ece(prediction_array, target_array),
        "num_examples": len(predictions),
    }


def _row_target(row: Mapping[str, Any]) -> float:
    if "expected_accuracy" in row:
        value = row["expected_accuracy"]
    elif "answer" in row:
        value = row["answer"]
    else:
        raise KeyError("Validation rows must contain 'expected_accuracy' or 'answer'.")
    return _validate_target(value, description="Validation target")


def materialize_predictions(
    rows: Sequence[Mapping[str, Any]],
    prompts: Sequence[str],
    responses: Sequence[str],
    invalid_policy: str,
    output_format: ConfidenceOutputFormat | Literal["auto"] = "auto",
) -> tuple[list[dict[str, Any]], list[float], list[float], int]:
    if not (len(rows) == len(prompts) == len(responses)):
        raise ValueError("Rows, prompts, and responses must have the same length.")
    records = []
    predictions = []
    targets = []
    invalid_count = 0
    for row, prompt, response in zip(rows, prompts, responses):
        target = _row_target(row)
        parsed, parse_error = parse_confidence(response, output_format=output_format)
        if parsed is None:
            invalid_count += 1
            if invalid_policy == "error":
                raise ValueError(
                    "Could not parse confidence for "
                    f"{row.get('example_id', row.get('_source_line'))}: {parse_error}\n{response}"
                )
            if invalid_policy == "zero":
                prediction = 0.0
            elif invalid_policy == "nan":
                prediction = float("nan")
            else:
                raise ValueError(f"Unknown invalid_policy: {invalid_policy}")
        else:
            prediction = parsed
        records.append(
            {
                "example_id": row.get("example_id"),
                "source_line": row.get("_source_line"),
                "query": row.get("query"),
                "expected_accuracy": target,
                "prompt": prompt,
                "response": response,
                "parsed_confidence": parsed,
                "prediction_for_metrics": prediction,
                "parse_error": parse_error,
            }
        )
        predictions.append(prediction)
        targets.append(target)

    if invalid_policy == "nan":
        filtered = [
            (prediction, target)
            for prediction, target in zip(predictions, targets)
            if math.isfinite(prediction)
        ]
        if not filtered and rows:
            raise ValueError("No valid parsed predictions.")
        if filtered:
            predictions, targets = [list(values) for values in zip(*filtered)]
        else:
            predictions, targets = [], []
    return records, predictions, targets, invalid_count


def _validation_prompts(
    valid_dataset: Sequence[Mapping[str, Any]],
    prompt_template: str | None,
) -> list[str]:
    prompts = []
    for row in valid_dataset:
        if "model_prompt" in row:
            prompts.append(str(row["model_prompt"]))
        elif "prompt" in row:
            prompts.append(str(row["prompt"]))
        elif prompt_template is not None and "query" in row:
            prompts.append(prompt_template.format(question=row["query"]))
        else:
            raise KeyError("Validation rows require 'prompt', or prompt_template plus 'query'.")
    return prompts


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    try:
        return str(completion.text)
    except AttributeError as error:
        raise TypeError("Completions must be strings or objects with a 'text' attribute.") from error


def validate_llm_rollout(
    llm_rollout: Any,
    valid_dataset: Sequence[Mapping[str, Any]],
    *,
    sampling_params: Mapping[str, Any] | None = None,
    batch_size: int | None = None,
    invalid_policy: str = "zero",
    prompt_template: str | None = None,
    output_format: ConfidenceOutputFormat = "answer_tags",
) -> dict[str, float | int]:
    """Generate one confidence per validation row and return aggregate metrics.

    The rollout object must expose VLLMServer.generate_completions. Training rows
    work directly because they contain prompt and expected_accuracy fields.
    """
    rows = list(valid_dataset)
    if not rows:
        raise ValueError(
            "valid_dataset is empty; check the configured validation dataset path "
            "and train/validation split sizes."
        )
    prompts = _validation_prompts(rows, prompt_template)
    resolved_sampling_params = dict(DEFAULT_SAMPLING_PARAMS)
    if sampling_params is not None:
        resolved_sampling_params.update(sampling_params)
    if resolved_sampling_params.get("n") != 1:
        raise ValueError("Validation requires sampling_params['n'] == 1.")

    if prompts:
        completions = llm_rollout.generate_completions(
            prompts=prompts,
            sampling_params=resolved_sampling_params,
            batch_size=batch_size,
        )
        responses = [_completion_text(completion) for completion in completions]
    else:
        raise ValueError("prompts shouldn't be empty")
    _, predictions, targets, invalid_count = materialize_predictions(
        rows,
        prompts,
        responses,
        invalid_policy,
        output_format,
    )
    metrics = compute_metrics(predictions, targets)
    metrics.update(
        {
            "num_invalid": invalid_count,
            "num_scored": len(predictions),
            "num_total": len(rows),
        }
    )
    return metrics


evaluate_llm_rollout = validate_llm_rollout


def infer_tensor_parallel_size(vllm_device: str, tensor_parallel_size: int | None) -> int:
    if tensor_parallel_size is not None:
        return tensor_parallel_size
    devices = [device.strip() for device in vllm_device.split(",") if device.strip()]
    return max(1, len(devices))


def prepare_model_for_vllm(
    model_folder: str,
    *,
    dtype: str,
    trust_remote_code: bool,
) -> tuple[str, Any | None]:
    """Merge a PEFT adapter into a temporary full model for vLLM.

    Full-model directories pass through unchanged. The returned temporary
    directory handle must stay alive until vLLM finishes using the model.
    """
    model_path = Path(model_folder)
    if not (model_path / "adapter_config.json").is_file():
        return str(model_path), None

    import gc
    import tempfile

    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name_or_path = peft_config.base_model_name_or_path
    if not base_model_name_or_path:
        raise ValueError(f"Adapter config has no base_model_name_or_path: {model_path}")

    print(
        f"Detected PEFT adapter at {model_path}; loading base model "
        f"{base_model_name_or_path} and merging before vLLM initialization."
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, model_path)
    merged_model = peft_model.merge_and_unload(safe_merge=True)

    merged_model_dir = tempfile.TemporaryDirectory(prefix="cc_merged_model_")
    merged_model.save_pretrained(merged_model_dir.name, safe_serialization=True)

    tokenizer_source = (
        model_path
        if (model_path / "tokenizer_config.json").is_file()
        else base_model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(merged_model_dir.name)

    del tokenizer, merged_model, peft_model, base_model
    gc.collect()
    return merged_model_dir.name, merged_model_dir


def initialize_vllm(args: argparse.Namespace) -> tuple[Any, Any]:
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
    return LLM(**llm_kwargs), sampling_params


def generate_with_vllm(llm: Any, sampling_params: Any, prompts: list[str]) -> list[str]:
    if not prompts:
        return []
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    for output in outputs:
        if len(output.outputs) != 1:
            raise RuntimeError(f"Expected one completion per prompt, got {len(output.outputs)}.")
    return [output.outputs[0].text for output in outputs]


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.model_folder = str(resolve_existing_path(args.model_folder, repo_root=repo_root))
    dataset_paths = resolve_existing_paths(args.test_data_jsonl, repo_root=repo_root)
    prompt_path = resolve_existing_path(args.prompt_path, repo_root=repo_root)
    if args.chat_template_path is not None:
        args.chat_template_path = str(
            resolve_existing_path(args.chat_template_path, repo_root=repo_root)
        )
    if args.chat_template_path is not None and not args.use_chat_template:
        raise ValueError("--chat_template_path requires --use_chat_template.")
    metrics_filename = validate_output_filename(
        args.output_metrics_jsonl,
        argument_name="--output_metrics_jsonl",
    )
    traces_filename = validate_output_filename(
        args.cc_prediction_traces_jsonl,
        argument_name="--cc_prediction_traces_jsonl",
    )
    metrics_paths = build_output_paths(dataset_paths, metrics_filename)
    traces_paths = build_output_paths(dataset_paths, traces_filename)
    prompt_template = load_prompt_template(prompt_path)
    print(f"Using prompt template {prompt_path}")
    prompt_tokenizer = None
    if args.use_chat_template:
        from transformers import AutoTokenizer

        prompt_tokenizer = AutoTokenizer.from_pretrained(
            args.model_folder,
            trust_remote_code=args.trust_remote_code,
        )
        configure_chat_template(
            prompt_tokenizer,
            use_chat_template=True,
            chat_template_path=args.chat_template_path,
        )
        template_description = args.chat_template_path or "saved with the model"
        print(f"Rendering prompts with chat template {template_description}")
    original_model_folder = args.model_folder
    merged_model_dir = None
    try:
        args.model_folder, merged_model_dir = prepare_model_for_vllm(
            args.model_folder,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        print(
            f"Using model {original_model_folder} on "
            f"CUDA_VISIBLE_DEVICES={args.vllm_device}"
        )
        llm, sampling_params = initialize_vllm(args)

        for dataset_index, (dataset_path, metrics_path, traces_path) in enumerate(
            zip(dataset_paths, metrics_paths, traces_paths),
            start=1,
        ):
            rows = load_dataset(
                dataset_path,
                seed=args.seed,
                shuffle=args.shuffle,
                start_index=args.start_index,
                limit=args.limit,
            )
            prompts = build_prompts(
                rows,
                prompt_template,
                tokenizer=prompt_tokenizer,
                use_chat_template=args.use_chat_template,
            )
            print(f"[{dataset_index}/{len(dataset_paths)}] Loaded {len(rows)} examples from {dataset_path}")
            responses = generate_with_vllm(llm, sampling_params, prompts)
            records, predictions, targets, invalid_count = materialize_predictions(
                rows,
                prompts,
                responses,
                args.invalid_policy,
                args.confidence_output_format,
            )
            metrics = compute_metrics(predictions, targets)
            metrics.update(
                {
                    "num_invalid": invalid_count,
                    "num_scored": len(predictions),
                    "num_total": len(rows),
                }
            )
            write_json(metrics_path, metrics)
            write_jsonl(traces_path, records)
            print(f"[{dataset_index}/{len(dataset_paths)}] Wrote metrics to {metrics_path}")
            print(f"[{dataset_index}/{len(dataset_paths)}] Wrote prediction traces to {traces_path}")
            print(json.dumps(metrics, indent=2, sort_keys=True))
            print(f"MSE: {metrics['mse']:.8f}")
            print(f"Spearman: {metrics['spearman_r']:.8f}")
    finally:
        if merged_model_dir is not None:
            merged_model_dir.cleanup()


if __name__ == "__main__":
    main()
