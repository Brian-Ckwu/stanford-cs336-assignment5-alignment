import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OLMo with GRPO on GSM8K.")

    parser.add_argument("--wandb-project-name", default="OLMo-2-0425-1B_GRPO_GSM8K")
    parser.add_argument(
        "--wandb-exp-name",
        default="r1-zero-prompt_default-hparams_max-tokens-256_lora-r-16-a-32-dropout-0-fp32",
    )
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--policy-device", type=int, default=2)
    parser.add_argument("--rollout-device", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--prompt-path", default="prompts/r1_zero.prompt")
    parser.add_argument("--add-cc-sft-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cc-sft-loss-lambda", type=float, default=1.0)
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--valid-batch-size", type=int, default=1024)
    parser.add_argument(
        "--difficulty-filter",
        choices=("none", "empirical-reward", "confidence-estimator"),
        default="none",
    )
    parser.add_argument(
        "--confidence-estimator-lora-dir",
        type=str,
        default=None,
        help=(
            "Required when --difficulty-filter is set to confidence-estimator. "
            "The directory must contain adapter_config.json and "
            "token_confidence_config.json."
        ),
    )
    parser.add_argument(
        "--confidence-estimator-batch-size",
        type=int,
        default=6400,
        help="Prompt batch size for confidence-estimator inference.",
    )
    parser.add_argument("--difficulty-lower-bound", type=float, default=0.2)
    parser.add_argument("--difficulty-upper-bound", type=float, default=0.8)
    parser.add_argument(
        "--max-candidate-groups-multiplier",
        type=int,
        default=10,
        help=(
            "Per-step candidate safety limit as a multiple of "
            "train_batch_size // group_size."
        ),
    )
    parser.add_argument(
        "--validation-n",
        type=int,
        default=1,
        help="Number of sampled completions per validation prompt.",
    )
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=256)
    parser.add_argument("--sampling-stop", default="</answer>")
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
    parser.add_argument(
        "--save-only-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save only the trained PEFT adapter in the final checkpoint instead "
            "of merging it into the base model. Requires --use-peft."
        ),
    )
    return parser.parse_args()


args = parse_args()
wandb_project_name = args.wandb_project_name
wandb_exp_name = args.wandb_exp_name
model_id = args.model_id
policy_device = args.policy_device
rollout_device = args.rollout_device
gpu_memory_utilization = args.gpu_memory_utilization
weight_transfer_backend = "ipc" if policy_device == rollout_device else "nccl"
prompt_path = args.prompt_path
n_train_examples = args.n_train_examples
n_val_examples = args.n_val_examples
num_rollout_steps = args.num_rollout_steps
learning_rate = args.learning_rate
vllm_max_num_seqs = args.vllm_max_num_seqs
max_num_batched_tokens = args.max_num_batched_tokens
train_batch_size = args.train_batch_size
group_size = args.group_size
gradient_accumulation_steps = args.gradient_accumulation_steps
sampling_temperature = args.sampling_temperature
sampling_max_tokens = args.sampling_max_tokens
max_grad_norm = args.max_grad_norm
adamw_betas = tuple(args.adamw_betas)
weight_decay = args.weight_decay
seed = args.seed
track_policy_memory = args.track_policy_memory
track_step_time = args.track_step_time
use_peft = args.use_peft
peft_method = args.peft_method
lora_r = args.lora_r
lora_alpha = args.lora_alpha
lora_dropout = args.lora_dropout
lora_adapter_name = args.lora_adapter_name
lora_target_modules = args.lora_target_modules
autocast_adapter_dtype = args.autocast_adapter_dtype
confidence_estimator_lora_dir = (
    None
    if args.confidence_estimator_lora_dir is None
    else Path(args.confidence_estimator_lora_dir).resolve()
)
confidence_estimator_batch_size = args.confidence_estimator_batch_size

if vllm_max_num_seqs <= 0:
    raise ValueError("--vllm-max-num-seqs must be positive")
if max_num_batched_tokens is not None and max_num_batched_tokens <= 0:
    raise ValueError("--max-num-batched-tokens must be positive")
if train_batch_size <= 0:
    raise ValueError("--train-batch-size must be positive")
if group_size <= 0:
    raise ValueError("--group-size must be positive")
if train_batch_size % group_size != 0:
    raise ValueError("--train-batch-size must be divisible by --group-size")
if args.max_candidate_groups_multiplier <= 0:
    raise ValueError("--max-candidate-groups-multiplier must be positive")
if args.save_only_adapter and not use_peft:
    raise ValueError("--save-only-adapter requires --use-peft")
if confidence_estimator_batch_size <= 0:
    raise ValueError("--confidence-estimator-batch-size must be positive")
if (
    args.difficulty_filter == "confidence-estimator"
    and confidence_estimator_lora_dir is None
):
    raise ValueError(
        "--confidence-estimator-lora-dir must be specified when "
        "--difficulty-filter is confidence-estimator"
    )
if args.difficulty_filter == "confidence-estimator" and not use_peft:
    raise ValueError("--difficulty-filter confidence-estimator requires --use-peft")

confidence_adapter_config = None
confidence_metadata = None
confidence_candidate_token_ids: tuple[int, ...] = ()
confidence_lora_rank = 0
if confidence_estimator_lora_dir is not None:
    adapter_config_path = confidence_estimator_lora_dir / "adapter_config.json"
    metadata_path = confidence_estimator_lora_dir / "token_confidence_config.json"
    for required_path in (adapter_config_path, metadata_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    with adapter_config_path.open() as handle:
        confidence_adapter_config = json.load(handle)
    with metadata_path.open() as handle:
        confidence_metadata = json.load(handle)
    confidence_base_model = confidence_adapter_config["base_model_name_or_path"]
    if confidence_base_model != model_id:
        raise ValueError(
            "Confidence estimator and policy must share the same base model: "
            f"{confidence_base_model!r} != {model_id!r}"
        )
    confidence_group_size = int(confidence_metadata["group_size"])
    if confidence_group_size != group_size:
        raise ValueError(
            "Confidence estimator group size does not match --group-size: "
            f"{confidence_group_size} != {group_size}"
        )
    confidence_candidate_token_ids = tuple(
        int(token_id) for token_id in confidence_metadata["candidate_token_ids"]
    )
    if len(confidence_candidate_token_ids) != group_size + 1:
        raise ValueError(
            "Confidence estimator must define one candidate token for each "
            "count from 0 through --group-size"
        )
    confidence_lora_rank = int(confidence_adapter_config["r"])

sampling_params = {
    "temperature": sampling_temperature,
    "max_tokens": sampling_max_tokens,
    "n": group_size,
    "seed": seed,
    "stop": args.sampling_stop,
    "include_stop_str_in_output": args.include_stop_str_in_output,
}
validation_sampling_params = sampling_params.copy()
if args.validation_n <= 0:
    raise ValueError("--validation-n must be positive.")
validation_sampling_params.update({"n": args.validation_n})

import os
from dotenv import load_dotenv
load_dotenv()
print(f"HF_HOME: {os.getenv('HF_HOME')}")

import wandb
wandb.login()
wandb_config = {
    "lr": learning_rate,
    "seed": seed,
    "difficulty_filter": args.difficulty_filter,
    "difficulty_lower_bound": args.difficulty_lower_bound,
    "difficulty_upper_bound": args.difficulty_upper_bound,
    "vllm_max_num_seqs": vllm_max_num_seqs,
    "max_num_batched_tokens": max_num_batched_tokens,
    "max_candidate_groups_multiplier": (
        args.max_candidate_groups_multiplier
    ),
    "confidence_estimator_lora_dir": (
        None
        if confidence_estimator_lora_dir is None
        else str(confidence_estimator_lora_dir)
    ),
    "confidence_estimator_batch_size": confidence_estimator_batch_size,
}
wandb_run = wandb.init(project=wandb_project_name, name=wandb_exp_name, config=wandb_config)

# Seeding
import torch
import random

# TODO: Set random seeds for numpy, torch, ...
random.seed(seed)

# Load dataset
with open(prompt_path) as f:
    prompt_template = f.read()

full_dataset = list()
with open("../data/gsm8k/train.jsonl") as f:
    for line in f:
        row = json.loads(line)
        row["prompt"] = prompt_template.replace("{question}", row["question"]).replace("{group_size}", str(group_size))  # NOTE: group_size in the prompt is meant for the cc loss
        row["answer"] = row["answer"].split("####")[-1].strip()
        full_dataset.append(row)
random.shuffle(full_dataset)
train_dataset = full_dataset[:n_train_examples]
valid_dataset = full_dataset[n_train_examples:n_train_examples+n_val_examples]

if len(train_dataset) < train_batch_size // group_size:
    raise ValueError(
        "Training dataset must contain at least "
        f"{train_batch_size // group_size} examples"
    )

print(f"Train dataset size: {len(train_dataset)}; validation dataset size: {len(valid_dataset)}")

# Load model copies (A: for updating the policy; B: for generating rollouts)
# A: policy model
from checkpoint import get_model_and_tokenizer

llm_policy_device = f"cuda:{policy_device}"
llm_policy, tokenizer = get_model_and_tokenizer(model_id, device=llm_policy_device)

from peft import LoraConfig, TaskType, get_peft_model
if use_peft:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
    )
    llm_policy = get_peft_model(llm_policy, peft_config, autocast_adapter_dtype=autocast_adapter_dtype)
    llm_policy.print_trainable_parameters()
    lora_dtype = {
        parameter.dtype
        for name, parameter in llm_policy.named_parameters()
        if "lora_" in name
    }
    print(f"LoRA dtype: {lora_dtype}")

optimizer = torch.optim.AdamW(
    (p for p in llm_policy.parameters() if p.requires_grad),
    lr=learning_rate,
    betas=adamw_betas,
    weight_decay=weight_decay
)

# B: rollout model
from vllm_utils import VLLMConfidenceEstimator, VLLMServer

confidence_filter_enabled = args.difficulty_filter == "confidence-estimator"
rollout_max_lora_rank = max(lora_r, confidence_lora_rank)

llm_rollout = VLLMServer(
    model_id=model_id,
    gpu=rollout_device,
    seed=seed,
    gpu_memory_utilization=gpu_memory_utilization,
    weight_transfer_backend=weight_transfer_backend,
    enable_lora=use_peft,
    max_lora_rank=rollout_max_lora_rank,
    max_loras=2 if confidence_filter_enabled else 1,
    max_num_seqs=vllm_max_num_seqs,
    max_num_batched_tokens=max_num_batched_tokens,
)
print(f"Starting the rollout model (vLLM service)...")
llm_rollout.start()
runtime_adapter_dir = None
if use_peft:
    import tempfile

    runtime_adapter_dir = tempfile.TemporaryDirectory(prefix="grpo_lora_")
    llm_policy.save_pretrained(runtime_adapter_dir.name, safe_serialization=True)
    llm_rollout.load_lora_adapter(
        lora_adapter_name,
        runtime_adapter_dir.name,
    )
else:
    llm_rollout.init_weight_sync(policy_device=llm_policy_device)  # NOTE: Create the communication channel between two llms

confidence_estimator = None
if confidence_filter_enabled:
    confidence_adapter_name = "confidence-estimator"
    llm_rollout.load_lora_adapter(
        confidence_adapter_name,
        str(confidence_estimator_lora_dir),
        set_default=False,
    )
    confidence_estimator = VLLMConfidenceEstimator(
        server=llm_rollout,
        adapter_name=confidence_adapter_name,
        candidate_token_ids=confidence_candidate_token_ids,
        group_size=group_size,
    )

# Training loop
from grpo_core_implementation import grpo_train_step, track_cuda_memory_and_time
from drgrpo_grader import r1_zero_reward_fn
from rollout_batch_collection import (
    ConfidenceEstimatorRolloutBatchCollector,
    EmpiricalRewardRolloutBatchCollector,
)

n_questions_per_train_batch = train_batch_size // group_size
print(
    f"vLLM max concurrent sequences: {vllm_max_num_seqs}; "
    f"# Questions per training batch: {n_questions_per_train_batch}; "
    f"# Generations per question: {group_size}"
)

filtered_batch_collector = None
if args.difficulty_filter == "empirical-reward":
    filtered_batch_collector = EmpiricalRewardRolloutBatchCollector(
        rollout_server=llm_rollout,
        train_rows=train_dataset,
        reward_fn=r1_zero_reward_fn,
        sampling_params=sampling_params,
        train_batch_size=train_batch_size,
        group_size=group_size,
        lower_bound=args.difficulty_lower_bound,
        upper_bound=args.difficulty_upper_bound,
        max_candidate_groups=(
            args.max_candidate_groups_multiplier
            * n_questions_per_train_batch
        ),
        scheduler_seed=seed,
    )
elif args.difficulty_filter == "confidence-estimator":
    filtered_batch_collector = ConfidenceEstimatorRolloutBatchCollector(
        rollout_server=llm_rollout,
        train_rows=train_dataset,
        reward_fn=r1_zero_reward_fn,
        confidence_estimator=confidence_estimator,
        sampling_params=sampling_params,
        train_batch_size=train_batch_size,
        group_size=group_size,
        lower_bound=args.difficulty_lower_bound,
        upper_bound=args.difficulty_upper_bound,
        max_candidate_groups=(
            args.max_candidate_groups_multiplier
            * n_questions_per_train_batch
        ),
        confidence_batch_size=confidence_estimator_batch_size,
        scheduler_seed=seed,
    )
next_unfiltered_train_index = 0

from tqdm import tqdm

# Validation before training
from evaluation import evaluate
validation_metrics = evaluate(
    vllm_server=llm_rollout,
    eval_dataset=valid_dataset,
    sampling_params=validation_sampling_params,
    batch_size=args.valid_batch_size,
    reward_fn=r1_zero_reward_fn,
    tokenizer=tokenizer if args.add_cc_sft_loss else None,
    cc_group_size=group_size if args.add_cc_sft_loss else None,
)
wandb_run.log(data={
    **{f"valid/{key}": value for key, value in validation_metrics.items()}
}, step=0)

for i in tqdm(range(num_rollout_steps), desc="GRPO training steps"):
    time_metrics = {}
    collection_metrics = {}
    precomputed_reward_dicts = None
    collected_batch_for_recording = None
    with track_cuda_memory_and_time(
        "rollout_full_batch",
        time_metrics=time_metrics,
        track_memory=False,
        track_time=track_step_time,
    ):
        if filtered_batch_collector is not None:
            collected_batch = filtered_batch_collector.collect()
            collected_batch_for_recording = collected_batch
            prompts = collected_batch.repeated_prompts
            responses = collected_batch.rollout_responses
            answers = collected_batch.repeated_ground_truths
            precomputed_reward_dicts = collected_batch.reward_dicts
            collection_metrics = collected_batch.metrics
        else:
            train_row_indices = [
                (
                    next_unfiltered_train_index + offset
                ) % len(train_dataset)
                for offset in range(n_questions_per_train_batch)
            ]
            train_rows = [
                train_dataset[index] for index in train_row_indices
            ]
            next_unfiltered_train_index = (
                next_unfiltered_train_index
                + n_questions_per_train_batch
            ) % len(train_dataset)
            print(
                f"Generating rollouts for the following {len(train_rows)} "
                "questions (answers): ",
                [
                    train_row["question"].split()[0]
                    + f" ({train_row['answer']})"
                    for train_row in train_rows
                ],
            )
            vllm_prompts, prompts, answers = list(), list(), list()
            for train_row in train_rows:
                vllm_prompts.append(train_row["prompt"])
                prompts.extend([train_row["prompt"]] * group_size)
                answers.extend([train_row["answer"]] * group_size)
            step_sampling_params = dict(sampling_params)
            step_sampling_params["seed"] = (
                seed + i * n_questions_per_train_batch
            )
            completions = llm_rollout.generate_completions(
                prompts=vllm_prompts,
                sampling_params=step_sampling_params,
                batch_size=None,
            )
            responses = [completion.text for completion in completions]

    assert len(prompts) == len(responses) == len(answers) == train_batch_size
    print(
        f"Successfully collected {len(responses)} training rollouts from "
        f"{len(prompts) // group_size} questions!"
    )
    # Print out sampled generations every 10 steps
    if i % 10 == 0:
        print(f"Prompt: {prompts[0]}")
        for index in range(group_size):
            print(f"Response: {responses[index]}")
    # A single train step
    llm_policy.train()
    with track_cuda_memory_and_time(
        "policy_full_batch",
        device=next(llm_policy.parameters()).device,
        time_metrics=time_metrics,
        track_memory=False,
        track_time=track_step_time,
    ):
        train_step_loss, train_step_metadata = grpo_train_step(
            model=llm_policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            reward_fn=r1_zero_reward_fn,
            repeated_prompts=prompts,
            rollout_responses=responses,
            repeated_ground_truths=answers,
            group_size=group_size,
            track_policy_memory=track_policy_memory,
            track_step_time=track_step_time,
            add_cc_sft_loss=args.add_cc_sft_loss,
            cc_sft_loss_lambda=args.cc_sft_loss_lambda,
            precomputed_reward_dicts=precomputed_reward_dicts,
        )
    if collected_batch_for_recording is not None:
        filtered_batch_collector.record_trained(
            collected_batch_for_recording
        )
    # Sync weights
    print("Syncing weights of the rollout LLM to be the same with the updated policy LLM...")
    sync_memory_metrics = {}
    with track_cuda_memory_and_time(
        "weight_sync",
        device=next(llm_policy.parameters()).device,
        memory_metrics=sync_memory_metrics,
        time_metrics=time_metrics,
        time_name="policy_rollout_weight_sync",
        track_memory=track_policy_memory,
        track_time=track_step_time,
    ):
        if use_peft:
            llm_policy.save_pretrained(
                runtime_adapter_dir.name,
                safe_serialization=True,
            )
            llm_rollout.load_lora_adapter(
                lora_adapter_name,
                runtime_adapter_dir.name,
                load_inplace=True,
            )
        else:
            llm_rollout.sync_policy_weights(policy=llm_policy)
    memory_metrics = train_step_metadata.pop("memory_metrics")
    memory_metrics.update(sync_memory_metrics)
    time_metrics.update(train_step_metadata.pop("time_metrics"))
    print("Syncing done!")
    wandb_run.log(data={
        "train/loss": train_step_loss,
        **{f"train/{key}": value for key, value in train_step_metadata.items()},
        **{
            (
                key
                if key.startswith("calibration/")
                else f"train/filter/{key}"
            ): value
            for key, value in collection_metrics.items()
        },
        **memory_metrics,
        **time_metrics,
    }, step=i)
    # Validation
    if ((i + 1) % 10 == 0) or (i == num_rollout_steps - 1):
        validation_metrics = evaluate(
            vllm_server=llm_rollout,
            eval_dataset=valid_dataset,
            sampling_params=validation_sampling_params,
            batch_size=args.valid_batch_size,
            reward_fn=r1_zero_reward_fn,
            tokenizer=tokenizer if args.add_cc_sft_loss else None,
            cc_group_size=group_size if args.add_cc_sft_loss else None,
        )
        wandb_run.log(data={
            **{f"valid/{key}": value for key, value in validation_metrics.items()}
        }, step=i)

# Closing
llm_rollout.stop()
if runtime_adapter_dir is not None:
    runtime_adapter_dir.cleanup()

output_dir = f"checkpoints/{wandb_exp_name}-final"
tokenizer.save_pretrained(output_dir)
if use_peft and args.save_only_adapter:
    llm_policy.save_pretrained(output_dir, safe_serialization=True)
elif use_peft:
    merged = llm_policy.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
else:
    llm_policy.save_pretrained(output_dir, safe_serialization=True)

wandb_run.finish()
