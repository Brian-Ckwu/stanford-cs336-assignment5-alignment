#!/usr/bin/env python3
"""Train a LoRA next-token confidence estimator on frozen policy rollouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import torch
from dotenv import load_dotenv

from .confidence_training.artifacts import RunArtifacts
from .confidence_training.config import (
    DEFAULT_LORA_TARGET_MODULES,
    TokenConfidenceConfig,
)
from .confidence_training.data import (
    ConfidenceDataset,
    PromptCollator,
    hash_prompts,
    load_rollout_confidence_data,
    preflight_prompt_lengths,
)
from .confidence_training.model import (
    build_optimizer,
    load_token_confidence_model,
    parameter_report,
)
from .confidence_training.trainer import set_global_seed, train_run
from .confidence_training.wandb_logging import WandbLogger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--model-revision")
    parser.add_argument("--output-dir", required=True)
    head_group = parser.add_mutually_exclusive_group()
    head_group.add_argument("--use-classification-head", action="store_true")
    head_group.add_argument("--use-regression-head", action="store_true")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--validate-every-k-steps", type=int, default=10)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--adamw-betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--adamw-epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=DEFAULT_LORA_TARGET_MODULES,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TokenConfidenceConfig:
    prediction_mode = "vocab"
    if args.use_classification_head:
        prediction_mode = "classification"
    elif args.use_regression_head:
        prediction_mode = "regression"
    config = TokenConfidenceConfig(
        run_name=args.run_name,
        data_dir=str(Path(args.data_dir).resolve()),
        model_id=args.model_id,
        model_revision=args.model_revision,
        output_dir=str(Path(args.output_dir).resolve()),
        prediction_mode=prediction_mode,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        max_sequence_length=args.max_sequence_length,
        validate_every_k_steps=args.validate_every_k_steps,
        log_every_steps=args.log_every_steps,
        max_grad_norm=args.max_grad_norm,
        adamw_betas=tuple(args.adamw_betas),
        adamw_epsilon=args.adamw_epsilon,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(args.lora_target_modules),
        gradient_checkpointing=args.gradient_checkpointing,
        trust_remote_code=args.trust_remote_code,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        resume=args.resume,
    )
    config.validate()
    return config


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def main(argv: Sequence[str] | None = None) -> None:
    package_root = Path(__file__).resolve().parent
    load_dotenv(package_root / ".env")
    configured_hf_home = os.environ.get("HF_HOME")
    if configured_hf_home and not Path(configured_hf_home).is_absolute():
        os.environ["HF_HOME"] = str((package_root / configured_hf_home).resolve())
    args = parse_args(argv)
    config = config_from_args(args)
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {config.device}")

    set_global_seed(config.seed)
    data = load_rollout_confidence_data(config.data_dir, group_size=config.group_size)
    model, tokenizer, lora_matches = load_token_confidence_model(config)
    length_report = {
        "train": preflight_prompt_lengths(
            tokenizer,
            data.train,
            max_sequence_length=config.max_sequence_length,
        ),
        "validation": preflight_prompt_lengths(
            tokenizer,
            data.validation,
            max_sequence_length=config.max_sequence_length,
        ),
    }
    optimizer = build_optimizer(model, config)
    parameters = parameter_report(model, optimizer)
    if not parameters["optimizer_matches_trainable"]:
        raise RuntimeError("Optimizer membership does not match trainable parameters")
    expected_trainable_kinds = (
        {"adapter"}
        if config.prediction_mode == "vocab"
        else {"adapter", "head"}
    )
    if set(parameters["trainable_by_kind"]) != expected_trainable_kinds:
        raise RuntimeError(
            "Unexpected trainable parameter groups: "
            f"expected {expected_trainable_kinds}, got {parameters['trainable_by_kind']}"
        )

    provenance = {
        "git_commit": _git_commit(),
        "collection_manifest": data.manifest,
        "source_hashes": data.source_hashes,
        "train_prompt_hash": hash_prompts(data.train),
        "validation_prompt_hash": hash_prompts(data.validation),
        "prompt_lengths": length_report,
        "candidate_tokens": (
            {
                str(count): int(token_id)
                for count, token_id in enumerate(model.candidate_token_ids.tolist())
            }
            if config.prediction_mode == "vocab"
            else None
        ),
        "resolved_model_revision": getattr(
            getattr(model.backbone, "config", None), "_commit_hash", config.model_revision
        ),
        "prediction_mode": config.prediction_mode,
        "objective": {
            "vocab": "full_vocabulary_next_token_cross_entropy",
            "classification": "class_cross_entropy",
            "regression": "binary_cross_entropy_with_logits",
        }[config.prediction_mode],
        "prediction_rule": {
            "vocab": "argmax_over_count_tokens_0_through_group_size",
            "classification": "argmax_over_count_classes_0_through_group_size",
            "regression": "sigmoid_of_scalar_logit",
        }[config.prediction_mode],
        "parameter_report": parameters,
        "lora_target_matches": lora_matches,
        "versions": {
            package: _package_version(package)
            for package in ("torch", "transformers", "peft", "wandb")
        },
    }
    artifacts = RunArtifacts(config)
    artifacts.initialize(provenance)
    tokenizer_path = artifacts.root / "tokenizer"
    if not tokenizer_path.exists():
        tokenizer.save_pretrained(tokenizer_path)

    train_collate = PromptCollator(
        tokenizer,
        device=config.device,
        max_sequence_length=config.max_sequence_length,
    )
    validation_collate = PromptCollator(
        tokenizer,
        device=config.device,
        max_sequence_length=config.max_sequence_length,
    )
    logger = WandbLogger.create(config)
    callback = (
        (lambda values, step: logger.log(values, step=step))
        if logger is not None
        else None
    )
    try:
        manifest = train_run(
            model=model,
            optimizer=optimizer,
            train_dataset=ConfidenceDataset(data.train),
            validation_dataset=ConfidenceDataset(data.validation),
            train_collate=train_collate,
            validation_collate=validation_collate,
            config=config,
            artifacts=artifacts,
            log_callback=callback,
        )
    except Exception as error:
        artifacts.mark_failed(str(error))
        raise
    finally:
        if logger is not None:
            logger.finish()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
