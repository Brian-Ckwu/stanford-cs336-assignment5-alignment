"""Confidence estimators backed by a causal vocabulary or a new prediction head."""

from __future__ import annotations

import json
from collections import defaultdict
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PredictionMode, TokenConfidenceConfig


def confidence_token_ids(tokenizer: Any, group_size: int) -> tuple[int, ...]:
    """Return the unique one-token encodings of counts ``0..group_size``."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    token_ids: list[int] = []
    for count in range(group_size + 1):
        encoded = tokenizer.encode(str(count), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Expected confidence count {count} to encode as one token, got {encoded}"
            )
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"Confidence counts do not map to unique tokens: {token_ids}")
    return tuple(token_ids)


def last_non_padding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be two-dimensional")
    if torch.any(attention_mask.sum(dim=1) <= 0):
        raise ValueError("Every prompt must contain a non-padding token")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return (
        positions.expand_as(attention_mask)
        .masked_fill(attention_mask == 0, -1)
        .max(1)
        .values
    )


def _backbone_config(backbone: nn.Module) -> Any:
    config = getattr(backbone, "config", None)
    if config is None and hasattr(backbone, "get_base_model"):
        config = getattr(backbone.get_base_model(), "config", None)
    if config is None:
        raise ValueError("Backbone must expose a model configuration")
    return config


def _pool_last_hidden_state(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    indices = last_non_padding_indices(attention_mask)
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, indices].float()


class ConfidenceModel(nn.Module):
    """Predict confidence using the LM vocabulary, a class head, or a scalar head."""

    def __init__(
        self,
        backbone: nn.Module,
        candidate_token_ids: Sequence[int] | None = None,
        *,
        prediction_mode: PredictionMode = "vocab",
        group_size: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.prediction_mode = prediction_mode
        if group_size is None:
            if candidate_token_ids is None:
                raise ValueError("group_size is required without candidate token IDs")
            group_size = len(candidate_token_ids) - 1
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.group_size = group_size

        candidate_ids = tuple(candidate_token_ids or ())
        if prediction_mode == "vocab" and len(candidate_ids) != group_size + 1:
            raise ValueError("Vocabulary mode requires one candidate token per count")
        self.register_buffer(
            "candidate_token_ids",
            torch.tensor(candidate_ids, dtype=torch.long),
            persistent=True,
        )

        self.prediction_head: nn.Linear | None = None
        if prediction_mode != "vocab":
            hidden_size = getattr(_backbone_config(backbone), "hidden_size", None)
            if hidden_size is None:
                raise ValueError("Backbone configuration must define hidden_size")
            output_size = group_size + 1 if prediction_mode == "classification" else 1
            self.prediction_head = nn.Linear(
                int(hidden_size), output_size, dtype=torch.float32
            )
            nn.init.xavier_uniform_(self.prediction_head.weight)
            nn.init.zeros_(self.prediction_head.bias)

    def forward_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )
        indices = last_non_padding_indices(attention_mask)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        if self.prediction_mode == "vocab":
            return outputs.logits[rows, indices].float()
        if self.prediction_head is None:
            raise RuntimeError("Prediction head is missing")
        pooled = _pool_last_hidden_state(outputs.last_hidden_state, attention_mask)
        scores = self.prediction_head(pooled)
        return scores.squeeze(-1) if self.prediction_mode == "regression" else scores

    def forward_next_token_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Backward-compatible vocabulary-mode forward method."""

        if self.prediction_mode != "vocab":
            raise ValueError("Next-token logits exist only in vocabulary mode")
        return self.forward_scores(input_ids, attention_mask, **kwargs)

    def target_token_ids(self, target_counts: torch.Tensor) -> torch.Tensor:
        if self.prediction_mode != "vocab":
            raise ValueError("Target token IDs exist only in vocabulary mode")
        if target_counts.dtype != torch.long:
            target_counts = target_counts.long()
        if torch.any(target_counts < 0) or torch.any(target_counts > self.group_size):
            raise ValueError("Target confidence count is outside the configured range")
        return self.candidate_token_ids[target_counts]

    def compute_loss(
        self,
        scores: torch.Tensor,
        target_counts: torch.Tensor,
        target_accuracies: torch.Tensor | None = None,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if self.prediction_mode == "vocab":
            return F.cross_entropy(
                scores.float(), self.target_token_ids(target_counts), reduction=reduction
            )
        if self.prediction_mode == "classification":
            return F.cross_entropy(scores.float(), target_counts.long(), reduction=reduction)
        if target_accuracies is None:
            raise ValueError("Regression mode requires target_accuracies")
        return F.binary_cross_entropy_with_logits(
            scores.float(), target_accuracies.float(), reduction=reduction
        )

    def categorical_logits(self, scores: torch.Tensor) -> torch.Tensor:
        if self.prediction_mode == "vocab":
            return scores.index_select(1, self.candidate_token_ids)
        if self.prediction_mode == "classification":
            return scores
        raise ValueError("Regression mode does not have categorical logits")

    def candidate_logits(self, scores: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for vocabulary/classification logits."""

        return self.categorical_logits(scores)

    def predict_counts(self, scores: torch.Tensor) -> torch.Tensor:
        return self.categorical_logits(scores).argmax(dim=1)

    def expected_counts(self, scores: torch.Tensor) -> torch.Tensor:
        probabilities = self.categorical_logits(scores).softmax(dim=1)
        counts = torch.arange(
            probabilities.shape[1],
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        return (probabilities * counts).sum(dim=1)

    def predict_confidences(self, scores: torch.Tensor) -> torch.Tensor:
        if self.prediction_mode == "regression":
            return torch.sigmoid(scores.float())
        return self.predict_counts(scores).float() / self.group_size

    def expected_confidences(self, scores: torch.Tensor) -> torch.Tensor | None:
        if self.prediction_mode == "regression":
            return None
        return self.expected_counts(scores) / self.group_size


# Preserve imports used by the first token-only implementation.
TokenConfidenceModel = ConfidenceModel


def _attention_implementation(device: torch.device) -> str:
    if device.type == "cpu":
        return "eager"
    if find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def load_confidence_model(
    config: TokenConfidenceConfig,
) -> tuple[ConfidenceModel, Any, dict[str, list[str]]]:
    """Load Qwen base weights and attach a new trainable BF16 LoRA and optional head."""

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    device = torch.device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define a pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    candidate_ids = (
        confidence_token_ids(tokenizer, config.group_size)
        if config.prediction_mode == "vocab"
        else ()
    )

    model_class = AutoModelForCausalLM if config.prediction_mode == "vocab" else AutoModel
    backbone = model_class.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
        dtype=torch.bfloat16,
        attn_implementation=_attention_implementation(device),
    )
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    if config.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()

    module_names = [name for name, _ in backbone.named_modules()]
    matches = {
        suffix: sorted(
            name
            for name in module_names
            if name == suffix or name.endswith(f".{suffix}")
        )
        for suffix in config.lora_target_modules
    }
    missing = [suffix for suffix, names in matches.items() if not names]
    if missing:
        raise ValueError(f"LoRA target modules matched nothing: {missing}")

    task_type = (
        TaskType.CAUSAL_LM
        if config.prediction_mode == "vocab"
        else TaskType.FEATURE_EXTRACTION
    )
    backbone = get_peft_model(
        backbone,
        LoraConfig(
            task_type=task_type,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
        ),
        autocast_adapter_dtype=False,
    )
    for name, parameter in backbone.named_parameters():
        if "lora_" in name and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(torch.bfloat16)
    model = ConfidenceModel(
        backbone,
        candidate_ids,
        prediction_mode=config.prediction_mode,
        group_size=config.group_size,
    ).to(device)
    return model, tokenizer, matches


def load_token_confidence_model(
    config: TokenConfidenceConfig,
) -> tuple[ConfidenceModel, Any, dict[str, list[str]]]:
    """Backward-compatible name for ``load_confidence_model``."""

    return load_confidence_model(config)


def build_optimizer(
    model: ConfidenceModel,
    config: TokenConfidenceConfig,
) -> torch.optim.AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=config.adamw_betas,
        eps=config.adamw_epsilon,
        weight_decay=config.weight_decay,
    )


def parameter_report(
    model: ConfidenceModel,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    dtypes: dict[str, int] = defaultdict(int)
    trainable_by_kind: dict[str, int] = defaultdict(int)
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        dtypes[str(parameter.dtype)] += count
        if parameter.requires_grad:
            trainable += count
            if "prediction_head" in name:
                kind = "head"
            elif "lora_" in name:
                kind = "adapter"
            else:
                kind = "other"
            trainable_by_kind[kind] += count
    duplicate_ids = {
        parameter_id
        for parameter_id in optimizer_ids
        if sum(id(parameter) == parameter_id for parameter in optimizer_parameters) != 1
    }
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100.0 * trainable / total,
        "trainable_by_kind": dict(sorted(trainable_by_kind.items())),
        "parameters_by_dtype": dict(sorted(dtypes.items())),
        "optimizer_matches_trainable": (
            optimizer_ids == trainable_ids and not duplicate_ids
        ),
        "optimizer_parameter_occurrences": len(optimizer_parameters),
        "optimizer_unique_parameter_tensors": len(optimizer_ids),
    }


def save_confidence_model(
    model: ConfidenceModel,
    destination: str | Path,
) -> None:
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=False)
    model.backbone.save_pretrained(path, safe_serialization=True)
    if model.prediction_head is not None:
        torch.save(model.prediction_head.state_dict(), path / "confidence_head.pt")
    config = _backbone_config(model.backbone)
    (path / "confidence_model_config.json").write_text(
        json.dumps(
            {
                "architecture": type(model).__name__,
                "prediction_mode": model.prediction_mode,
                "candidate_token_ids": model.candidate_token_ids.detach().cpu().tolist(),
                "group_size": model.group_size,
                "hidden_size": getattr(config, "hidden_size", None),
                "head_output_size": (
                    model.prediction_head.out_features
                    if model.prediction_head is not None
                    else None
                ),
                "objective": (
                    "binary_cross_entropy_with_logits"
                    if model.prediction_mode == "regression"
                    else "cross_entropy"
                ),
                "prediction_rule": (
                    "sigmoid"
                    if model.prediction_mode == "regression"
                    else "argmax_over_confidence_counts"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def save_token_confidence_model(
    model: ConfidenceModel,
    destination: str | Path,
) -> None:
    """Backward-compatible name for ``save_confidence_model``."""

    save_confidence_model(model, destination)


def load_confidence_checkpoint(
    source: str | Path,
    *,
    device: str | torch.device = "cpu",
    tokenizer_source: str | Path | None = None,
) -> tuple[ConfidenceModel, Any]:
    from peft import PeftConfig, PeftModel
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    path = Path(source)
    metadata_path = path / "confidence_model_config.json"
    if not metadata_path.exists():
        metadata_path = path / "token_confidence_config.json"
    metadata = json.loads(metadata_path.read_text())
    prediction_mode = metadata.get("prediction_mode", "vocab")
    peft_config = PeftConfig.from_pretrained(path)
    resolved_device = torch.device(device)
    model_class = AutoModelForCausalLM if prediction_mode == "vocab" else AutoModel
    base = model_class.from_pretrained(
        peft_config.base_model_name_or_path,
        revision=getattr(peft_config, "revision", None),
        dtype=torch.bfloat16,
        attn_implementation=_attention_implementation(resolved_device),
    )
    backbone = PeftModel.from_pretrained(
        base,
        path,
        is_trainable=False,
        autocast_adapter_dtype=False,
    )
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    model = ConfidenceModel(
        backbone,
        metadata.get("candidate_token_ids", ()),
        prediction_mode=prediction_mode,
        group_size=int(metadata["group_size"]),
    )
    head_path = path / "confidence_head.pt"
    if model.prediction_head is not None:
        if not head_path.is_file():
            raise FileNotFoundError(f"Checkpoint is missing its prediction head: {head_path}")
        model.prediction_head.load_state_dict(
            torch.load(head_path, map_location="cpu", weights_only=True)
        )
    model = model.to(resolved_device)
    if tokenizer_source is None:
        run_tokenizer = path.parents[2] / "tokenizer"
        tokenizer_source = (
            run_tokenizer if run_tokenizer.is_dir() else peft_config.base_model_name_or_path
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_token_confidence_checkpoint(
    source: str | Path,
    *,
    device: str | torch.device = "cpu",
    tokenizer_source: str | Path | None = None,
) -> tuple[ConfidenceModel, Any]:
    """Backward-compatible name for ``load_confidence_checkpoint``."""

    return load_confidence_checkpoint(
        source, device=device, tokenizer_source=tokenizer_source
    )
