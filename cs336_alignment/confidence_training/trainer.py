"""Training and validation loops for next-token confidence prediction."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .artifacts import RunArtifacts
from .config import TokenConfidenceConfig
from .data import ConfidenceDataset
from .metrics import compute_metrics
from .model import ConfidenceModel, save_confidence_model


@dataclass(frozen=True, slots=True)
class ValidationOutput:
    metrics: dict[str, float | int]
    predictions: tuple[dict[str, Any], ...]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def validate(
    model: ConfidenceModel,
    loader: DataLoader,
    *,
    group_size: int,
) -> ValidationOutput:
    was_training = model.training
    model.eval()
    records: list[dict[str, Any]] = []
    predictions: list[float] = []
    expected_predictions: list[float] = []
    targets: list[float] = []
    loss_sum = 0.0
    exact_correct = 0
    full_vocab_valid = 0
    try:
        for batch in loader:
            scores = model.forward_scores(**batch["inputs"])
            target_counts = batch["target_counts"]
            target_accuracies = batch["target_accuracies"]
            batch_loss = model.compute_loss(
                scores,
                target_counts,
                target_accuracies,
                reduction="none",
            )
            loss_sum += float(batch_loss.sum())

            if model.prediction_mode == "regression":
                batch_predictions = model.predict_confidences(scores)
                predicted_counts = None
                expected_confidences = None
                full_argmax = None
                valid_full_argmax = None
                target_token_ids = None
                target_probabilities = None
            else:
                categorical_probabilities = model.categorical_logits(scores).softmax(dim=1)
                predicted_counts = categorical_probabilities.argmax(dim=1)
                batch_predictions = predicted_counts.float() / group_size
                expected_confidences = model.expected_confidences(scores)
                exact_correct += int(predicted_counts.eq(target_counts).sum())
                if model.prediction_mode == "vocab":
                    target_token_ids = model.target_token_ids(target_counts)
                    full_argmax = scores.argmax(dim=1)
                    valid_full_argmax = full_argmax[:, None].eq(
                        model.candidate_token_ids[None, :]
                    ).any(dim=1)
                    full_vocab_valid += int(valid_full_argmax.sum())
                    target_probabilities = scores.log_softmax(dim=1).gather(
                        1, target_token_ids[:, None]
                    ).squeeze(1).exp()
                else:
                    target_token_ids = None
                    full_argmax = None
                    valid_full_argmax = None
                    target_probabilities = categorical_probabilities.gather(
                        1, target_counts[:, None]
                    ).squeeze(1)

            for index, example_id in enumerate(batch["example_ids"]):
                target_count = int(target_counts[index])
                target_confidence = float(target_accuracies[index])
                predicted_confidence = float(batch_predictions[index])
                predictions.append(predicted_confidence)
                targets.append(target_confidence)
                record: dict[str, Any] = {
                    "example_id": example_id,
                    "prediction_mode": model.prediction_mode,
                    "target_count": target_count,
                    "target_confidence": target_confidence,
                    "predicted_confidence": predicted_confidence,
                    "loss": float(batch_loss[index]),
                }
                if predicted_counts is not None and expected_confidences is not None:
                    expected_confidence = float(expected_confidences[index])
                    expected_predictions.append(expected_confidence)
                    record.update(
                        {
                            "predicted_count": int(predicted_counts[index]),
                            "expected_confidence": expected_confidence,
                            "target_class_probability": float(target_probabilities[index]),
                        }
                    )
                if model.prediction_mode == "vocab":
                    assert target_token_ids is not None
                    assert full_argmax is not None
                    assert valid_full_argmax is not None
                    record.update(
                        {
                            "target_token_id": int(target_token_ids[index]),
                            "full_vocab_argmax_token_id": int(full_argmax[index]),
                            "full_vocab_argmax_is_valid": bool(valid_full_argmax[index]),
                        }
                    )
                records.append(record)
    finally:
        model.train(was_training)

    primary_metrics = compute_metrics(predictions, targets)
    total = len(records)
    metrics: dict[str, float | int] = {
        **primary_metrics,
        "validation_loss": loss_sum / total,
    }
    if model.prediction_mode == "regression":
        metrics["bce"] = loss_sum / total
    else:
        expected_metrics = compute_metrics(expected_predictions, targets)
        metrics.update(
            {
                "nll": loss_sum / total,
                "count_accuracy": exact_correct / total,
                **{
                    f"expected_{key}": value
                    for key, value in expected_metrics.items()
                    if key != "num_examples"
                },
            }
        )
        if model.prediction_mode == "vocab":
            metrics["token_accuracy"] = sum(
                row["full_vocab_argmax_token_id"] == row["target_token_id"]
                for row in records
            ) / total
            metrics["full_vocab_valid_token_rate"] = full_vocab_valid / total
        else:
            metrics["class_accuracy"] = exact_correct / total
    return ValidationOutput(metrics=metrics, predictions=tuple(records))


def _save_resume_state(
    path: Path,
    *,
    epoch: int,
    optimizer_steps: int,
    model: ConfidenceModel,
    optimizer: torch.optim.Optimizer,
    config: TokenConfidenceConfig,
) -> None:
    from peft import get_peft_model_state_dict

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "resume_state_version": 1,
            "scientific_config_hash": config.scientific_hash(),
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "adapter": {
                name: value.detach().cpu().clone()
                for name, value in get_peft_model_state_dict(model.backbone).items()
            },
            "prediction_head": (
                None
                if model.prediction_head is None
                else {
                    name: value.detach().cpu().clone()
                    for name, value in model.prediction_head.state_dict().items()
                }
            ),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        temporary,
    )
    os.replace(temporary, path)


def _load_resume_state(
    path: Path,
    *,
    model: ConfidenceModel,
    optimizer: torch.optim.Optimizer,
    config: TokenConfidenceConfig,
) -> tuple[int, int]:
    from peft import set_peft_model_state_dict

    payload = torch.load(path, map_location=config.device, weights_only=False)
    if payload["scientific_config_hash"] != config.scientific_hash():
        raise ValueError("Resume state configuration does not match")
    set_peft_model_state_dict(model.backbone, payload["adapter"])
    saved_head = payload.get("prediction_head")
    if model.prediction_head is None:
        if saved_head is not None:
            raise ValueError("Resume state contains a head for vocabulary mode")
    else:
        if saved_head is None:
            raise ValueError("Resume state is missing its prediction head")
        model.prediction_head.load_state_dict(saved_head)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return int(payload["epoch"]), int(payload["optimizer_steps"])


def train_run(
    *,
    model: ConfidenceModel,
    optimizer: torch.optim.Optimizer,
    train_dataset: ConfidenceDataset,
    validation_dataset: ConfidenceDataset,
    train_collate: Callable,
    validation_collate: Callable,
    config: TokenConfidenceConfig,
    artifacts: RunArtifacts,
    log_callback: Callable[[dict[str, Any], int], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(config.device)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=validation_collate,
    )
    start_epoch = 0
    optimizer_steps = 0
    if config.resume:
        if not artifacts.resume_path.is_file():
            raise FileNotFoundError(f"Missing resume state: {artifacts.resume_path}")
        start_epoch, optimizer_steps = _load_resume_state(
            artifacts.resume_path,
            model=model,
            optimizer=optimizer,
            config=config,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if start_epoch == 0:
        initial = validate(model, validation_loader, group_size=config.group_size)
        row = {
            "kind": "validation",
            "epoch": 0,
            "optimizer_steps": 0,
            **initial.metrics,
        }
        artifacts.append_history(row)
        if log_callback:
            log_callback({f"validation/{key}": value for key, value in initial.metrics.items()}, 0)

    for epoch in range(start_epoch + 1, config.epochs + 1):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
            num_workers=0,
            collate_fn=train_collate,
        )
        model.train()
        epoch_loss_sum = 0.0
        epoch_examples = 0
        last_validation: ValidationOutput | None = None
        last_validation_step: int | None = None
        progress = tqdm(train_loader, desc=f"Confidence epoch {epoch}/{config.epochs}")
        for epoch_step, batch in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            scores = model.forward_scores(**batch["inputs"])
            loss = model.compute_loss(
                scores,
                batch["target_counts"],
                batch["target_accuracies"],
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config.max_grad_norm,
            )
            optimizer.step()
            optimizer_steps += 1
            batch_size = len(batch["example_ids"])
            epoch_examples += batch_size
            epoch_loss_sum += float(loss.detach()) * batch_size
            progress.set_postfix(loss=f"{float(loss.detach()):.5f}")

            if epoch_step % config.log_every_steps == 0 or epoch_step == len(train_loader):
                row = {
                    "kind": "train_step",
                    "epoch": epoch,
                    "epoch_step": epoch_step,
                    "optimizer_steps": optimizer_steps,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm.detach()),
                    "examples_seen_in_epoch": epoch_examples,
                }
                artifacts.append_history(row)
                if log_callback:
                    log_callback(
                        {
                            "train/loss": row["loss"],
                            "train/gradient_norm": row["gradient_norm"],
                            "train/epoch": epoch,
                        },
                        optimizer_steps,
                    )

            if optimizer_steps % config.validate_every_k_steps == 0:
                last_validation = validate(
                    model, validation_loader, group_size=config.group_size
                )
                last_validation_step = optimizer_steps
                row = {
                    "kind": "validation",
                    "epoch": epoch,
                    "optimizer_steps": optimizer_steps,
                    **last_validation.metrics,
                }
                artifacts.append_history(row)
                if log_callback:
                    log_callback(
                        {
                            f"validation/{key}": value
                            for key, value in last_validation.metrics.items()
                        },
                        optimizer_steps,
                    )

        if last_validation is None or last_validation_step != optimizer_steps:
            last_validation = validate(
                model, validation_loader, group_size=config.group_size
            )
            last_validation_step = optimizer_steps
        epoch_row = {
            "kind": "epoch",
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "train_loss": epoch_loss_sum / epoch_examples,
            **{
                f"validation_{key}": value
                for key, value in last_validation.metrics.items()
            },
        }
        artifacts.append_history(epoch_row)
        improved = artifacts.improvements(last_validation.metrics)
        artifacts.save_improved_checkpoint(
            epoch=epoch,
            optimizer_steps=optimizer_steps,
            metrics=last_validation.metrics,
            predictions=last_validation.predictions,
            improved=improved,
            model_saver=lambda path: save_confidence_model(model, path),
        )
        if log_callback:
            selectors = json.loads(artifacts.selector_path.read_text())
            log_callback(
                {
                    "best/mse_epoch": selectors["mse"]["epoch"],
                    "best/spearman_epoch": selectors["spearman_r"]["epoch"],
                    "best/mse": selectors["mse"]["value"],
                    "best/spearman": selectors["spearman_r"]["value"],
                },
                optimizer_steps,
            )
        _save_resume_state(
            artifacts.resume_path,
            epoch=epoch,
            optimizer_steps=optimizer_steps,
            model=model,
            optimizer=optimizer,
            config=config,
        )

    runtime = {
        "wall_seconds": time.perf_counter() - started,
        "optimizer_steps": optimizer_steps,
        "peak_vram_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "peak_vram_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        ),
    }
    artifacts.update_runtime(runtime)
    artifacts.mark_complete(trained_through_epoch=config.epochs)
    return json.loads(artifacts.manifest_path.read_text())
