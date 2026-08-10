from __future__ import annotations

import argparse
import gc
import json
import os
import random
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from .calibration_utils import configure_chat_template, render_user_prompt
    from .checkpoint import get_model_and_tokenizer
    from .drgrpo_grader import cc_reward_fn
    from .evaluation_with_cc import validate_llm_rollout
    from .grpo_core_implementation import grpo_train_step, track_cuda_memory_and_time
    from .vllm_utils import VLLMServer
except ImportError:  # Support `uv run training_script_grpo_on_cc.py`.
    from calibration_utils import configure_chat_template, render_user_prompt
    from checkpoint import get_model_and_tokenizer
    from drgrpo_grader import cc_reward_fn
    from evaluation_with_cc import validate_llm_rollout
    from grpo_core_implementation import grpo_train_step, track_cuda_memory_and_time
    from vllm_utils import VLLMServer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LLM with GRPO on Capability Calibration.")

    parser.add_argument("--wandb-project-name", default="GRPO_CC_TriviaQA")
    parser.add_argument(
        "--wandb-exp-name",
        default="Qwen/Qwen3-1.7B_lr-3e-5_lora-r-16-a-32-dropout-0-fp32",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--policy-device", type=int, default=2)
    parser.add_argument("--rollout-device", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
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
    parser.add_argument(
        "--confidence-output-format",
        choices=("answer_tags", "boxed"),
        default="answer_tags",
    )

    parser.add_argument("--n-train-examples", type=int, default=12800)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--sampling-temperature", type=float, default=0.6)
    parser.add_argument("--sampling-max-tokens", type=int, default=256)
    parser.add_argument("--sequence-max-tokens", type=int, default=1024)
    stop_group = parser.add_mutually_exclusive_group()
    stop_group.add_argument(
        "--sampling-stop",
        default=None,
        help="Stop string. Defaults to </answer> for answer_tags and no stop for boxed.",
    )
    stop_group.add_argument(
        "--no-sampling-stop",
        action="store_true",
        help="Disable stop-string handling regardless of output format.",
    )
    parser.add_argument(
        "--include-stop-str-in-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--lora-adapter-name", default="policy")
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


def resolve_sampling_stop(args: argparse.Namespace) -> str | None:
    if args.no_sampling_stop:
        return None
    if args.sampling_stop is not None:
        return args.sampling_stop
    if args.confidence_output_format == "answer_tags":
        return "</answer>"
    return None


def validate_args(args: argparse.Namespace) -> None:
    if args.chat_template_path is not None and not args.use_chat_template:
        raise ValueError("--chat-template-path requires --use-chat-template.")
    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError("--rollout-batch-size must be divisible by --group-size.")
    if args.gradient_accumulation_steps > args.train_batch_size:
        raise ValueError("--gradient-accumulation-steps cannot exceed --train-batch-size.")


def load_cc_dataset(
    dataset_path: str | Path,
    prompt_template: str,
) -> list[dict[str, Any]]:
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
            row["user_prompt"] = prompt_template.format(question=row["query"])
            row["answer"] = row["expected_accuracy"]
            rows.append(row)
    return rows


def render_dataset_prompts(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    use_chat_template: bool,
) -> list[dict[str, Any]]:
    rendered_rows = []
    for original_row in rows:
        row = dict(original_row)
        model_prompt = render_user_prompt(
            tokenizer,
            str(row["user_prompt"]),
            use_chat_template=use_chat_template,
        )
        row["model_prompt"] = model_prompt
        # Compatibility with validation helpers and existing trace consumers.
        row["prompt"] = model_prompt
        rendered_rows.append(row)
    return rendered_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    sampling_stop = resolve_sampling_stop(args)
    sampling_params = {
        "temperature": args.sampling_temperature,
        "top_p": 0.95,
        "max_tokens": args.sampling_max_tokens,
        "n": args.group_size,
        "seed": args.seed,
        "stop": sampling_stop,
        "include_stop_str_in_output": args.include_stop_str_in_output,
    }

    load_dotenv()
    print(f"HF_HOME: {os.getenv('HF_HOME')}")

    import wandb

    wandb.login()
    wandb_run = wandb.init(
        project=args.wandb_project_name,
        name=args.wandb_exp_name,
        config={
            "lr": args.learning_rate,
            "seed": args.seed,
            "use_chat_template": args.use_chat_template,
            "chat_template_path": args.chat_template_path,
            "confidence_output_format": args.confidence_output_format,
            "sampling_stop": sampling_stop,
        },
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    prompt_template = Path(args.prompt_path).read_text().strip()
    if "{question}" not in prompt_template:
        raise ValueError(f"Prompt template must contain '{{question}}': {args.prompt_path}")
    full_dataset = load_cc_dataset(args.dataset_path, prompt_template)
    random.shuffle(full_dataset)
    train_dataset = full_dataset[: args.n_train_examples]
    valid_dataset = full_dataset[
        args.n_train_examples : args.n_train_examples + args.n_val_examples
    ]
    print(
        f"Train dataset size: {len(train_dataset)}; "
        f"validation dataset size: {len(valid_dataset)}"
    )

    llm_policy_device = f"cuda:{args.policy_device}"
    llm_policy, tokenizer = get_model_and_tokenizer(
        args.model_id,
        device=llm_policy_device,
    )
    configure_chat_template(
        tokenizer,
        use_chat_template=args.use_chat_template,
        chat_template_path=args.chat_template_path,
    )
    train_dataset = render_dataset_prompts(
        train_dataset,
        tokenizer,
        use_chat_template=args.use_chat_template,
    )
    valid_dataset = render_dataset_prompts(
        valid_dataset,
        tokenizer,
        use_chat_template=args.use_chat_template,
    )

    from peft import LoraConfig, TaskType, get_peft_model

    if args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
        )
        llm_policy = get_peft_model(
            llm_policy,
            peft_config,
            autocast_adapter_dtype=args.autocast_adapter_dtype,
        )
        llm_policy.print_trainable_parameters()
        lora_dtype = {
            parameter.dtype
            for name, parameter in llm_policy.named_parameters()
            if "lora_" in name
        }
        print(f"LoRA dtype: {lora_dtype}")
    else:
        print("Running the experiment with full-weight fine-tuning...")

    optimizer = torch.optim.AdamW(
        (parameter for parameter in llm_policy.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=tuple(args.adamw_betas),
        weight_decay=args.weight_decay,
    )

    weight_transfer_backend = (
        "ipc" if args.policy_device == args.rollout_device else "nccl"
    )
    llm_rollout = VLLMServer(
        model_id=args.model_id,
        gpu=args.rollout_device,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        weight_transfer_backend=weight_transfer_backend,
        enable_lora=args.use_peft,
        max_lora_rank=args.lora_r,
        max_loras=1,
        max_model_len=args.sequence_max_tokens,
    )
    print("Starting the rollout model (vLLM service)...")
    llm_rollout.start()

    runtime_adapter_dir = None
    try:
        if args.use_peft:
            runtime_adapter_dir = tempfile.TemporaryDirectory(prefix="grpo_lora_")
            llm_policy.save_pretrained(
                runtime_adapter_dir.name,
                safe_serialization=True,
            )
            llm_rollout.load_lora_adapter(
                args.lora_adapter_name,
                runtime_adapter_dir.name,
            )
        else:
            llm_rollout.init_weight_sync(policy_device=llm_policy_device)

        assert (
            args.num_rollout_steps * args.rollout_batch_size
            == len(train_dataset) * args.group_size
        ), "Configured rollout steps must cover the shuffled training set exactly once."
        n_questions_per_rollout = args.rollout_batch_size // args.group_size
        print(
            f"Rollout batch size: {args.rollout_batch_size}; "
            f"# Questions per rollout: {n_questions_per_rollout}; "
            f"# Generations per question: {args.group_size}"
        )

        validation_sampling_params = {
            **sampling_params,
            "temperature": 0.6,
            "top_p": 0.95,
            "n": 1,
        }
        print("Validating the initial LLM policy...")
        valid_metrics = validate_llm_rollout(
            llm_rollout,
            valid_dataset,
            sampling_params=validation_sampling_params,
            batch_size=args.rollout_batch_size,
            output_format=args.confidence_output_format,
        )
        print(f"Validation metrics at step 0: {json.dumps(valid_metrics, sort_keys=True)}")
        wandb_run.log(
            data={
                "valid/mse": valid_metrics["mse"],
                "valid/spearman_r": valid_metrics["spearman_r"],
                "valid/num_invalid": valid_metrics["num_invalid"],
                "valid/num_total": valid_metrics["num_total"],
            },
            step=0,
        )
        max_spearman = valid_metrics["spearman_r"]
        reward_fn = partial(
            cc_reward_fn,
            output_format=args.confidence_output_format,
        )

        for step in tqdm(range(args.num_rollout_steps), desc="GRPO training steps"):
            time_metrics: dict[str, float] = {}
            start = step * n_questions_per_rollout
            train_rows = train_dataset[start : start + n_questions_per_rollout]
            print(
                f"Generating rollouts for {len(train_rows)} questions (answers): ",
                [
                    row["query"].split()[0] + f" ({row['answer']})"
                    for row in train_rows
                ],
            )

            vllm_prompts = [row["model_prompt"] for row in train_rows]
            prompts = [
                row["model_prompt"]
                for row in train_rows
                for _ in range(args.group_size)
            ]
            answers = [
                row["answer"]
                for row in train_rows
                for _ in range(args.group_size)
            ]
            assert len(prompts) == len(answers) == args.rollout_batch_size

            print(f"Generating {len(prompts)} rollouts with vLLM...")
            with track_cuda_memory_and_time(
                "rollout_full_batch",
                time_metrics=time_metrics,
                track_memory=False,
                track_time=args.track_step_time,
            ):
                completions = llm_rollout.generate_completions(
                    prompts=vllm_prompts,
                    sampling_params=sampling_params,
                    batch_size=args.rollout_batch_size,
                )
                responses = [completion.text for completion in completions]
                assert len(prompts) == len(responses)
            print(
                f"Successfully generated {len(responses)} rollouts for "
                f"{len(vllm_prompts)} prompts!"
            )

            if step % 10 == 0:
                print(f"Model prompt: {prompts[0]}")
                for index in range(args.group_size):
                    print(f"Response: {responses[index]}")

            llm_policy.train()
            with track_cuda_memory_and_time(
                "policy_full_batch",
                device=next(llm_policy.parameters()).device,
                time_metrics=time_metrics,
                track_memory=False,
                track_time=args.track_step_time,
            ):
                train_step_loss, train_step_metadata = grpo_train_step(
                    model=llm_policy,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    max_grad_norm=args.max_grad_norm,
                    reward_fn=reward_fn,
                    repeated_prompts=prompts,
                    rollout_responses=responses,
                    repeated_ground_truths=answers,
                    group_size=args.group_size,
                    track_policy_memory=args.track_policy_memory,
                    track_step_time=args.track_step_time,
                )
            garbage_count = gc.collect()
            torch.cuda.empty_cache()
            print(f"Memory released! ({garbage_count} garbages collected)")

            print("Syncing updated policy weights to the rollout LLM...")
            sync_memory_metrics: dict[str, float] = {}
            with track_cuda_memory_and_time(
                "weight_sync",
                device=next(llm_policy.parameters()).device,
                memory_metrics=sync_memory_metrics,
                time_metrics=time_metrics,
                time_name="policy_rollout_weight_sync",
                track_memory=args.track_policy_memory,
                track_time=args.track_step_time,
            ):
                if args.use_peft:
                    llm_policy.save_pretrained(
                        runtime_adapter_dir.name,
                        safe_serialization=True,
                    )
                    llm_rollout.load_lora_adapter(
                        args.lora_adapter_name,
                        runtime_adapter_dir.name,
                        load_inplace=True,
                    )
                else:
                    llm_rollout.sync_policy_weights(policy=llm_policy)
            memory_metrics = train_step_metadata.pop("memory_metrics")
            memory_metrics.update(sync_memory_metrics)
            time_metrics.update(train_step_metadata.pop("time_metrics"))
            print("Syncing done!")

            if (step + 1) % 10 == 0:
                print(f"Validating the LLM policy at step {step + 1}...")
                valid_metrics = validate_llm_rollout(
                    llm_rollout,
                    valid_dataset,
                    sampling_params=validation_sampling_params,
                    batch_size=args.rollout_batch_size,
                    output_format=args.confidence_output_format,
                )
                print(
                    f"Validation metrics at step {step + 1}: "
                    f"{json.dumps(valid_metrics, sort_keys=True)}"
                )
                if valid_metrics["spearman_r"] > max_spearman:
                    print(
                        "Current Spearman corr better than previous best: "
                        f"{valid_metrics['spearman_r']:.4f} > {max_spearman:.4f}"
                    )
                    max_spearman = valid_metrics["spearman_r"]
                    output_dir = f"checkpoints/{args.wandb_exp_name}-best-spearman"
                    tokenizer.save_pretrained(output_dir)
                    llm_policy.save_pretrained(output_dir, safe_serialization=True)
                wandb_run.log(
                    data={
                        "valid/mse": valid_metrics["mse"],
                        "valid/spearman_r": valid_metrics["spearman_r"],
                        "valid/num_invalid": valid_metrics["num_invalid"],
                        "valid/num_total": valid_metrics["num_total"],
                    },
                    step=step,
                    commit=False,
                )

            wandb_run.log(
                data={
                    "train/loss": train_step_loss,
                    **{
                        f"train/{key}": value
                        for key, value in train_step_metadata.items()
                    },
                    **memory_metrics,
                    **time_metrics,
                },
                step=step,
            )

        output_dir = f"checkpoints/{args.wandb_exp_name}-final"
        tokenizer.save_pretrained(output_dir)
        llm_policy.save_pretrained(output_dir, safe_serialization=True)
    finally:
        llm_rollout.stop()
        if runtime_adapter_dir is not None:
            runtime_adapter_dir.cleanup()
        wandb_run.finish()


if __name__ == "__main__":
    main()
