"""Typed configuration for the token confidence estimator."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


CONFIG_SCHEMA_VERSION = 1
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
PredictionMode = Literal["vocab", "classification", "regression"]
PREDICTION_MODES = ("vocab", "classification", "regression")


@dataclass(frozen=True, slots=True)
class TokenConfidenceConfig:
    """Fully resolved configuration for one offline-confidence run."""

    run_name: str
    data_dir: str
    model_id: str
    output_dir: str
    model_revision: str | None = None
    prediction_mode: PredictionMode = "vocab"
    group_size: int = 8
    learning_rate: float = 3e-5
    batch_size: int = 16
    validation_batch_size: int = 16
    epochs: int = 3
    seed: int = 42
    device: str = "cuda:0"
    max_sequence_length: int = 1024
    validate_every_k_steps: int = 10
    log_every_steps: int = 10
    max_grad_norm: float = 1.0
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_epsilon: float = 1e-8
    weight_decay: float = 0.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES
    gradient_checkpointing: bool = False
    trust_remote_code: bool = True
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled"
    resume: bool = False
    schema_version: int = CONFIG_SCHEMA_VERSION

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["adamw_betas"] = list(self.adamw_betas)
        value["lora_target_modules"] = list(self.lora_target_modules)
        return value

    def scientific_dict(self) -> dict[str, Any]:
        value = self.to_dict()
        for key in ("output_dir", "resume", "wandb_project", "wandb_entity", "wandb_mode"):
            value.pop(key, None)
        return value

    def scientific_hash(self) -> str:
        payload = json.dumps(
            self.scientific_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def validate(self, *, require_paths: bool = True) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported configuration schema: {self.schema_version}")
        if self.prediction_mode not in PREDICTION_MODES:
            raise ValueError(f"Unknown prediction_mode: {self.prediction_mode}")
        if require_paths:
            for name in ("run_name", "data_dir", "model_id", "output_dir"):
                if not getattr(self, name):
                    raise ValueError(f"{name} is required")
        for name, value in (
            ("group_size", self.group_size),
            ("batch_size", self.batch_size),
            ("validation_batch_size", self.validation_batch_size),
            ("epochs", self.epochs),
            ("max_sequence_length", self.max_sequence_length),
            ("validate_every_k_steps", self.validate_every_k_steps),
            ("log_every_steps", self.log_every_steps),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("adamw_epsilon", self.adamw_epsilon),
            ("max_grad_norm", self.max_grad_norm),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if len(self.adamw_betas) != 2 or not all(0.0 <= value < 1.0 for value in self.adamw_betas):
            raise ValueError("adamw_betas must contain two values in [0, 1)")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.lora_target_modules:
            raise ValueError("At least one LoRA target module is required")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError(f"Unknown W&B mode: {self.wandb_mode}")
        if self.wandb_mode != "disabled" and not self.wandb_project:
            raise ValueError("wandb_project is required when W&B is enabled")
