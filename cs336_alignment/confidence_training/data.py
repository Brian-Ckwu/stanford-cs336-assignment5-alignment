"""Strict adapter for offline policy-rollout confidence data."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset


TRAIN_LABELS = "aggregated_labels_train.jsonl"
VALIDATION_LABELS = "aggregated_labels_validation.jsonl"
SAMPLES = "generation_samples.jsonl"
MANIFEST = "manifest.json"


@dataclass(frozen=True, slots=True)
class ConfidenceExample:
    example_id: str
    prompt: str
    target_count: int
    group_size: int
    expected_accuracy: float
    split: str
    split_index: int


@dataclass(frozen=True, slots=True)
class RolloutConfidenceData:
    train: tuple[ConfidenceExample, ...]
    validation: tuple[ConfidenceExample, ...]
    manifest: dict[str, Any]
    source_hashes: dict[str, str]


class ConfidenceDataset(Dataset[ConfidenceExample]):
    def __init__(self, examples: Sequence[ConfidenceExample]) -> None:
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ConfidenceExample:
        return self.examples[index]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_prompts(examples: Sequence[ConfidenceExample]) -> str:
    payload = [
        {"example_id": example.example_id, "prompt": example.prompt}
        for example in examples
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield line_number, value


@dataclass(frozen=True, slots=True)
class _Label:
    example_id: str
    split: str
    split_index: int
    num_samples: int
    num_correct: int
    expected_accuracy: float
    checkpoint_id: str
    method: str


def _load_labels(path: Path, *, expected_split: str, group_size: int) -> list[_Label]:
    labels: list[_Label] = []
    seen: set[str] = set()
    for line_number, row in _iter_jsonl(path):
        example_id = str(row.get("example_id", ""))
        if not example_id:
            raise ValueError(f"{path}:{line_number} is missing example_id")
        if example_id in seen:
            raise ValueError(f"Duplicate example_id in {path}: {example_id}")
        seen.add(example_id)
        split = str(row.get("split", ""))
        if split != expected_split:
            raise ValueError(
                f"{path}:{line_number} has split={split!r}; expected {expected_split!r}"
            )
        num_samples = int(row["num_samples"])
        num_correct = int(row["num_correct"])
        expected_accuracy = float(row["expected_accuracy"])
        if num_samples != group_size:
            raise ValueError(
                f"{example_id} has num_samples={num_samples}; expected {group_size}"
            )
        if not 0 <= num_correct <= group_size:
            raise ValueError(f"Invalid num_correct={num_correct} for {example_id}")
        exact = num_correct / group_size
        if not math.isfinite(expected_accuracy) or not math.isclose(
            expected_accuracy, exact, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"expected_accuracy mismatch for {example_id}: {expected_accuracy} != {exact}"
            )
        labels.append(
            _Label(
                example_id=example_id,
                split=split,
                split_index=int(row["split_index"]),
                num_samples=num_samples,
                num_correct=num_correct,
                expected_accuracy=expected_accuracy,
                checkpoint_id=str(row.get("checkpoint_id", "")),
                method=str(row.get("method", "")),
            )
        )
    if not labels:
        raise ValueError(f"Dataset is empty: {path}")
    expected_indices = list(range(len(labels)))
    actual_indices = [label.split_index for label in labels]
    if actual_indices != expected_indices:
        raise ValueError(f"{path} split_index values must be ordered and contiguous from zero")
    return labels


def _load_prompts(
    path: Path,
    *,
    required_ids: set[str],
    group_size: int,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    prompts: dict[str, str] = {}
    checkpoint_ids: dict[str, str] = {}
    methods: dict[str, str] = {}
    sample_ids: dict[str, set[int]] = {example_id: set() for example_id in required_ids}
    for line_number, row in _iter_jsonl(path):
        example_id = str(row.get("example_id", ""))
        if example_id not in required_ids:
            continue
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{path}:{line_number} has an invalid prompt")
        if example_id in prompts and prompts[example_id] != prompt:
            raise ValueError(f"Samples for {example_id} do not share one exact prompt")
        prompts[example_id] = prompt
        sample_id = int(row["sample_id"])
        if sample_id in sample_ids[example_id]:
            raise ValueError(f"Duplicate sample_id={sample_id} for {example_id}")
        sample_ids[example_id].add(sample_id)
        checkpoint_id = str(row.get("checkpoint_id", ""))
        method = str(row.get("method", ""))
        if example_id in checkpoint_ids and checkpoint_ids[example_id] != checkpoint_id:
            raise ValueError(f"Samples for {example_id} have different checkpoint IDs")
        if example_id in methods and methods[example_id] != method:
            raise ValueError(f"Samples for {example_id} have different methods")
        checkpoint_ids[example_id] = checkpoint_id
        methods[example_id] = method
    missing = sorted(required_ids - set(prompts))
    if missing:
        raise ValueError(f"Missing generation prompts for IDs: {missing[:10]}")
    expected_sample_ids = set(range(group_size))
    malformed = [
        example_id
        for example_id, values in sample_ids.items()
        if values != expected_sample_ids
    ]
    if malformed:
        raise ValueError(
            f"Expected sample IDs 0..{group_size - 1}; malformed IDs: {malformed[:10]}"
        )
    return prompts, checkpoint_ids, methods


def load_rollout_confidence_data(
    root: str | Path,
    *,
    group_size: int = 8,
) -> RolloutConfidenceData:
    """Load fixed train/validation labels and exact prompts from one collection."""

    data_root = Path(root).resolve()
    paths = {
        "train_labels": data_root / TRAIN_LABELS,
        "validation_labels": data_root / VALIDATION_LABELS,
        "samples": data_root / SAMPLES,
        "manifest": data_root / MANIFEST,
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing offline-confidence artifacts: {missing_files}")
    manifest = json.loads(paths["manifest"].read_text())
    if int(manifest.get("sampling", {}).get("n", -1)) != group_size:
        raise ValueError("Collection manifest group size does not match training group size")

    train_labels = _load_labels(
        paths["train_labels"], expected_split="train", group_size=group_size
    )
    validation_labels = _load_labels(
        paths["validation_labels"], expected_split="validation", group_size=group_size
    )
    train_ids = {label.example_id for label in train_labels}
    validation_ids = {label.example_id for label in validation_labels}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise ValueError(f"Train/validation example IDs overlap: {overlap[:10]}")
    required_ids = train_ids | validation_ids
    prompts, prompt_checkpoints, prompt_methods = _load_prompts(
        paths["samples"], required_ids=required_ids, group_size=group_size
    )

    manifest_checkpoint = str(manifest.get("checkpoint_id", ""))
    manifest_method = str(manifest.get("method", ""))

    def materialize(labels: Sequence[_Label]) -> tuple[ConfidenceExample, ...]:
        examples: list[ConfidenceExample] = []
        for label in labels:
            if not label.checkpoint_id or label.checkpoint_id != manifest_checkpoint:
                raise ValueError(f"Checkpoint mismatch for {label.example_id}")
            if not label.method or label.method != manifest_method:
                raise ValueError(f"Collection-method mismatch for {label.example_id}")
            if prompt_checkpoints[label.example_id] != label.checkpoint_id:
                raise ValueError(f"Sample/label checkpoint mismatch for {label.example_id}")
            if prompt_methods[label.example_id] != label.method:
                raise ValueError(f"Sample/label method mismatch for {label.example_id}")
            examples.append(
                ConfidenceExample(
                    example_id=label.example_id,
                    prompt=prompts[label.example_id],
                    target_count=label.num_correct,
                    group_size=label.num_samples,
                    expected_accuracy=label.expected_accuracy,
                    split=label.split,
                    split_index=label.split_index,
                )
            )
        return tuple(examples)

    train = materialize(train_labels)
    validation = materialize(validation_labels)
    return RolloutConfidenceData(
        train=train,
        validation=validation,
        manifest=manifest,
        source_hashes={name: sha256_file(path) for name, path in paths.items()},
    )


class PromptCollator:
    """Tokenize exact policy prompts without chat re-rendering or truncation."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        device: torch.device | str,
        max_sequence_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.max_sequence_length = max_sequence_length

    def __call__(self, rows: Sequence[ConfidenceExample]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [row.prompt for row in rows],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        lengths = encoded["attention_mask"].sum(dim=1)
        longest = int(lengths.max())
        if longest > self.max_sequence_length:
            raise ValueError(
                f"Prompt length {longest} exceeds max_sequence_length="
                f"{self.max_sequence_length}; truncation is disabled"
            )
        inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        return {
            "inputs": inputs,
            "target_counts": torch.tensor(
                [row.target_count for row in rows],
                dtype=torch.long,
                device=self.device,
            ),
            "target_accuracies": torch.tensor(
                [row.expected_accuracy for row in rows],
                dtype=torch.float32,
                device=self.device,
            ),
            "example_ids": [row.example_id for row in rows],
        }


def preflight_prompt_lengths(
    tokenizer: Any,
    examples: Sequence[ConfidenceExample],
    *,
    max_sequence_length: int,
    batch_size: int = 256,
) -> dict[str, int]:
    """Check every prompt before creating output artifacts."""

    minimum: int | None = None
    maximum = 0
    for start in range(0, len(examples), batch_size):
        encoded = tokenizer(
            [row.prompt for row in examples[start : start + batch_size]],
            padding=False,
            truncation=False,
        )
        lengths = [len(values) for values in encoded["input_ids"]]
        minimum = min(lengths) if minimum is None else min(minimum, *lengths)
        maximum = max(maximum, *lengths)
    if maximum > max_sequence_length:
        raise ValueError(
            f"Prompt length {maximum} exceeds max_sequence_length={max_sequence_length}; "
            "truncation is disabled"
        )
    return {"minimum_prompt_tokens": minimum or 0, "maximum_prompt_tokens": maximum}
