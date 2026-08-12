"""Train a supervised capability-calibration regression baseline.

The model encodes the same prompt used by the generative CC policy, pools the
last non-padding token, and maps that representation to a confidence in [0, 1].
The backbone can be fine-tuned in full or through a LoRA adapter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from .calibration_utils import configure_chat_template, render_user_prompt
    from .evaluation_with_cc import compute_metrics
    from .grpo_core_implementation import track_cuda_memory_and_time
except ImportError:  # Support direct execution from cs336_alignment/.
    from calibration_utils import configure_chat_template, render_user_prompt
    from evaluation_with_cc import compute_metrics
    from grpo_core_implementation import track_cuda_memory_and_time


REGRESSION_CONFIG_FILENAME = "regression_config.json"
REGRESSION_HEAD_FILENAME = "regression_head.pt"
BACKBONE_DIRECTORY = "backbone"


def _backbone_hidden_size(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None and hasattr(backbone, "get_base_model"):
        hidden_size = getattr(backbone.get_base_model().config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("The backbone config must define hidden_size.")
    return int(hidden_size)


class RegressionWithLLMBackbone(nn.Module):
    """Decoder-only LLM with a scalar regression head on its final token."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        # MSE on probabilities benefits from fp32; casting one vector is cheap.
        self.regression_head = nn.Linear(
            _backbone_hidden_size(backbone),
            1,
            dtype=torch.float32,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is None:
            last_token_indices = torch.full(
                (batch_size,),
                sequence_length - 1,
                dtype=torch.long,
                device=hidden_states.device,
            )
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask and input_ids must have the same shape.")
            if not torch.all(attention_mask.sum(dim=1) > 0):
                raise ValueError("Every sequence must contain a non-padding token.")
            positions = torch.arange(sequence_length, device=hidden_states.device)
            positions = positions.expand(batch_size, -1)
            last_token_indices = (
                positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
            )

        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        pooled = hidden_states[batch_indices, last_token_indices]
        logits = self.regression_head(pooled.float()).squeeze(-1)
        return torch.sigmoid(logits)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LLM regression baseline on Capability Calibration."
    )
    parser.add_argument("--wandb-project-name", default="SFT_CC_TriviaQA")
    parser.add_argument(
        "--wandb-exp-name",
        default="Qwen/Qwen3-1.7B_regression_lr-1e-5_lora-r-16-a-32",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--device", default="0", help="CUDA index, cuda:N, or cpu.")
    parser.add_argument("--prompt-path", default="prompts/verbalized_cc.prompt")
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render each calibration prompt as one user chat message.",
    )
    parser.add_argument(
        "--chat-template-path",
        default=None,
        help="Optional Jinja override. Requires --use-chat-template.",
    )

    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="Effective number of examples in one optimizer update.",
    )
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Number of microbatches used for each effective training batch.",
    )
    parser.add_argument("--sequence-max-tokens", type=int, default=1024)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--adamw-betas",
        type=float,
        nargs=2,
        metavar=("BETA1", "BETA2"),
        default=(0.9, 0.95),
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--track-policy-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--track-step-time",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--use-peft",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--peft-method", choices=("lora",), default="lora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    parser.add_argument(
        "--autocast-adapter-dtype",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fp32 LoRA weights; pass --no-autocast-adapter-dtype for bf16.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive_arguments = {
        "--n-train-examples": args.n_train_examples,
        "--n-val-examples": args.n_val_examples,
        "--num-train-epochs": args.num_train_epochs,
        "--train-batch-size": args.train_batch_size,
        "--validation-batch-size": args.validation_batch_size,
        "--gradient-accumulation-steps": args.gradient_accumulation_steps,
        "--sequence-max-tokens": args.sequence_max_tokens,
    }
    for name, value in positive_arguments.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.validation_interval < 0:
        raise ValueError("--validation-interval must be non-negative.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive.")
    if args.gradient_accumulation_steps > args.train_batch_size:
        raise ValueError("--gradient-accumulation-steps cannot exceed --train-batch-size.")
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "--train-batch-size must be divisible by --gradient-accumulation-steps."
        )
    if args.chat_template_path is not None and not args.use_chat_template:
        raise ValueError("--chat-template-path requires --use-chat-template.")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1).")


def _validate_target(value: Any, *, line_number: int) -> float:
    target = float(value)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError(
            f"expected_accuracy on dataset row {line_number} must be finite and in [0, 1], "
            f"got {value!r}."
        )
    return target


def load_cc_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(dataset_path).open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "query" not in row or "expected_accuracy" not in row:
                raise KeyError(
                    f"Dataset row {line_number} requires query and expected_accuracy fields."
                )
            row["query"] = str(row["query"])
            row["expected_accuracy"] = _validate_target(
                row["expected_accuracy"], line_number=line_number
            )
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    return rows


def render_dataset_prompts(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    prompt_template: str,
    *,
    use_chat_template: bool,
) -> list[dict[str, Any]]:
    rendered_rows = []
    for original_row in rows:
        row = dict(original_row)
        user_prompt = prompt_template.format(question=row["query"])
        row["model_prompt"] = render_user_prompt(
            tokenizer,
            user_prompt,
            use_chat_template=use_chat_template,
        )
        rendered_rows.append(row)
    return rendered_rows


def resolve_device(device: str | int) -> torch.device:
    value = str(device)
    if value.isdigit():
        value = f"cuda:{value}"
    resolved = torch.device(value)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {resolved}")
    return resolved


def get_attn_implementation(device: str | torch.device) -> str:
    from importlib.util import find_spec

    device_type = torch.device(device).type
    if device_type == "cpu":
        return "eager"
    if find_spec("flash_attn") is not None:
        return "flash_attention_2"
    if device_type == "cuda":
        return "sdpa"
    return "eager"


def resolve_input_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        package_relative = Path(__file__).resolve().parent / candidate
        if package_relative.is_file():
            return package_relative
    raise FileNotFoundError(f"Could not find file: {path}")


def ensure_padding_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define either a pad token or an EOS token.")
    tokenizer.pad_token = tokenizer.eos_token


def iter_batches(rows: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def split_microbatches(
    rows: Sequence[Mapping[str, Any]],
    gradient_accumulation_steps: int,
) -> Iterator[Sequence[Mapping[str, Any]]]:
    """Split a batch into at most N balanced, non-empty microbatches."""
    n_microbatches = min(len(rows), gradient_accumulation_steps)
    quotient, remainder = divmod(len(rows), n_microbatches)
    start = 0
    for index in range(n_microbatches):
        size = quotient + (index < remainder)
        yield rows[start : start + size]
        start += size


def tokenize_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    encoded = tokenizer(
        [str(row["model_prompt"]) for row in rows],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }
    targets = torch.tensor(
        [float(row["expected_accuracy"]) for row in rows],
        device=device,
        dtype=torch.float32,
    )
    return inputs, targets


def regression_train_step(
    model: RegressionWithLLMBackbone,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    rows: Sequence[Mapping[str, Any]],
    *,
    gradient_accumulation_steps: int,
    max_length: int,
    max_grad_norm: float,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Run one update whose gradients equal a full-batch mean-MSE update."""
    if not rows:
        raise ValueError("A training batch cannot be empty.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch_size = len(rows)
    summed_loss = torch.zeros((), dtype=torch.float32, device=device)
    prediction_sum = torch.zeros((), dtype=torch.float32, device=device)
    max_sequence_length = 0

    for microbatch in split_microbatches(rows, gradient_accumulation_steps):
        inputs, targets = tokenize_rows(
            microbatch,
            tokenizer,
            max_length=max_length,
            device=device,
        )
        max_sequence_length = max(max_sequence_length, inputs["input_ids"].shape[1])
        predictions = model(**inputs)
        microbatch_loss_sum = F.mse_loss(predictions, targets, reduction="sum")
        (microbatch_loss_sum / batch_size).backward()
        summed_loss += microbatch_loss_sum.detach()
        prediction_sum += predictions.detach().sum()

    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return float((summed_loss / batch_size).cpu()), {
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "prediction_mean": float((prediction_sum / batch_size).cpu()),
        "max_sequence_length": float(max_sequence_length),
    }


@torch.inference_mode()
def validate_regression_model(
    model: RegressionWithLLMBackbone,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Validation dataset cannot be empty.")
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    for batch in iter_batches(rows, batch_size):
        inputs, batch_targets = tokenize_rows(
            batch,
            tokenizer,
            max_length=max_length,
            device=device,
        )
        predictions.extend(model(**inputs).float().cpu().tolist())
        targets.extend(batch_targets.cpu().tolist())
    metrics = compute_metrics(predictions, targets)
    metrics.update(
        {
            "num_invalid": 0,
            "num_scored": len(predictions),
            "num_total": len(rows),
        }
    )
    return metrics


def save_regression_checkpoint(
    output_dir: str | Path,
    model: RegressionWithLLMBackbone,
    tokenizer: Any,
    *,
    model_id: str,
    use_peft: bool,
    training_args: Mapping[str, Any] | None = None,
) -> None:
    output_path = Path(output_dir)
    backbone_path = output_path / BACKBONE_DIRECTORY
    backbone_path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(backbone_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    torch.save(
        model.regression_head.state_dict(),
        output_path / REGRESSION_HEAD_FILENAME,
    )
    config: dict[str, Any] = {
        "architecture": type(model).__name__,
        "base_model_id": model_id,
        "backbone_directory": BACKBONE_DIRECTORY,
        "head_filename": REGRESSION_HEAD_FILENAME,
        "output_activation": "sigmoid",
        "pooling": "last_non_padding_token",
        "use_peft": use_peft,
    }
    if training_args is not None:
        config["training_args"] = dict(training_args)
    (output_path / REGRESSION_CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


def load_regression_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[RegressionWithLLMBackbone, Any]:
    """Reload a checkpoint produced by save_regression_checkpoint."""
    from transformers import AutoModel, AutoTokenizer

    checkpoint_path = Path(checkpoint_dir)
    config = json.loads((checkpoint_path / REGRESSION_CONFIG_FILENAME).read_text())
    backbone_path = checkpoint_path / config["backbone_directory"]
    resolved_device = torch.device(device)
    model_kwargs = {
        "device_map": str(resolved_device),
        "dtype": dtype,
        "attn_implementation": get_attn_implementation(resolved_device),
    }
    if config["use_peft"]:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(backbone_path)
        backbone = AutoModel.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        backbone = PeftModel.from_pretrained(backbone, backbone_path)
    else:
        backbone = AutoModel.from_pretrained(backbone_path, **model_kwargs)
    model = RegressionWithLLMBackbone(backbone).to(resolved_device)
    head_state = torch.load(
        checkpoint_path / config["head_filename"],
        map_location=resolved_device,
        weights_only=True,
    )
    model.regression_head.load_state_dict(head_state)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    ensure_padding_token(tokenizer)
    return model, tokenizer


def _wandb_validation_metrics(metrics: Mapping[str, float | int]) -> dict[str, Any]:
    return {f"valid/{name}": value for name, value in metrics.items()}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    load_dotenv()
    print(f"HF_HOME: {os.getenv('HF_HOME')}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    full_dataset = load_cc_dataset(args.dataset_path)
    required_examples = args.n_train_examples + args.n_val_examples
    if len(full_dataset) < required_examples:
        raise ValueError(
            f"Dataset contains {len(full_dataset)} rows, but {required_examples} are required "
            "by --n-train-examples plus --n-val-examples."
        )
    random.Random(args.seed).shuffle(full_dataset)
    train_dataset = full_dataset[: args.n_train_examples]
    valid_dataset = full_dataset[
        args.n_train_examples : args.n_train_examples + args.n_val_examples
    ]
    print(
        f"Train dataset size: {len(train_dataset)}; "
        f"validation dataset size: {len(valid_dataset)}"
    )

    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    backbone = AutoModel.from_pretrained(
        args.model_id,
        device_map=str(device),
        dtype=dtype,
        attn_implementation=get_attn_implementation(device),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    ensure_padding_token(tokenizer)
    configure_chat_template(
        tokenizer,
        use_chat_template=args.use_chat_template,
        chat_template_path=args.chat_template_path,
    )
    prompt_path = resolve_input_path(args.prompt_path)
    prompt_template = prompt_path.read_text().strip()
    if "{question}" not in prompt_template:
        raise ValueError(f"Prompt template must contain '{{question}}': {prompt_path}")
    train_dataset = render_dataset_prompts(
        train_dataset,
        tokenizer,
        prompt_template,
        use_chat_template=args.use_chat_template,
    )
    valid_dataset = render_dataset_prompts(
        valid_dataset,
        tokenizer,
        prompt_template,
        use_chat_template=args.use_chat_template,
    )

    if args.use_peft:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
        )
        backbone = get_peft_model(
            backbone,
            peft_config,
            autocast_adapter_dtype=args.autocast_adapter_dtype,
        )
        backbone.print_trainable_parameters()
        lora_dtypes = {
            parameter.dtype
            for name, parameter in backbone.named_parameters()
            if "lora_" in name
        }
        print(f"LoRA dtype: {lora_dtypes}")
    else:
        print("Running the experiment with full-weight fine-tuning...")

    # AutoModel can otherwise construct a KV cache that training never uses.
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()

    model = RegressionWithLLMBackbone(backbone).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=tuple(args.adamw_betas),
        weight_decay=args.weight_decay,
    )

    import wandb

    run_config = vars(args).copy()
    run_config["resolved_device"] = str(device)
    run_config["resolved_prompt_path"] = str(prompt_path)
    wandb_run = wandb.init(
        project=args.wandb_project_name,
        name=args.wandb_exp_name,
        config=run_config,
        mode=args.wandb_mode,
    )

    checkpoint_prefix = Path(args.output_dir) / args.wandb_exp_name
    best_spearman = float("-inf")
    last_validation_step = -1
    training_args = vars(args).copy()

    def validate_and_maybe_save(step: int, *, allow_checkpoint: bool) -> None:
        nonlocal best_spearman, last_validation_step
        metrics = validate_regression_model(
            model,
            tokenizer,
            valid_dataset,
            batch_size=args.validation_batch_size,
            max_length=args.sequence_max_tokens,
            device=device,
        )
        print(f"Validation metrics at step {step}: {json.dumps(metrics, sort_keys=True)}")
        wandb_run.log(_wandb_validation_metrics(metrics), step=step)
        spearman = float(metrics["spearman_r"])
        if allow_checkpoint and spearman > best_spearman:
            print(f"New best validation Spearman correlation: {spearman:.4f}")
            save_regression_checkpoint(
                f"{checkpoint_prefix}-best-spearman",
                model,
                tokenizer,
                model_id=args.model_id,
                use_peft=args.use_peft,
                training_args=training_args,
            )
        best_spearman = max(best_spearman, spearman)
        last_validation_step = step

    try:
        print("Validating the initial regression model...")
        validate_and_maybe_save(0, allow_checkpoint=False)
        global_step = 0
        for epoch in range(args.num_train_epochs):
            epoch_rows = list(train_dataset)
            random.Random(args.seed + epoch).shuffle(epoch_rows)
            progress = tqdm(
                iter_batches(epoch_rows, args.train_batch_size),
                total=math.ceil(len(epoch_rows) / args.train_batch_size),
                desc=f"Regression epoch {epoch + 1}",
            )
            for train_rows in progress:
                global_step += 1
                memory_metrics: dict[str, float] = {}
                time_metrics: dict[str, float] = {}
                with track_cuda_memory_and_time(
                    "train_step",
                    device=device,
                    memory_metrics=memory_metrics,
                    time_metrics=time_metrics,
                    track_memory=args.track_policy_memory,
                    track_time=args.track_step_time,
                ):
                    loss, train_metrics = regression_train_step(
                        model,
                        tokenizer,
                        optimizer,
                        train_rows,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        max_length=args.sequence_max_tokens,
                        max_grad_norm=args.max_grad_norm,
                        device=device,
                    )
                progress.set_postfix(loss=f"{loss:.5f}")
                wandb_run.log(
                    {
                        "train/loss": loss,
                        "train/epoch": epoch + 1,
                        **{f"train/{key}": value for key, value in train_metrics.items()},
                        **memory_metrics,
                        **time_metrics,
                    },
                    step=global_step,
                )
                if (
                    args.validation_interval > 0
                    and global_step % args.validation_interval == 0
                ):
                    validate_and_maybe_save(global_step, allow_checkpoint=True)

            if last_validation_step != global_step:
                validate_and_maybe_save(global_step, allow_checkpoint=True)

        save_regression_checkpoint(
            f"{checkpoint_prefix}-final",
            model,
            tokenizer,
            model_id=args.model_id,
            use_peft=args.use_peft,
            training_args=training_args,
        )
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
