import argparse

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

    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=256)
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
rollout_batch_size = args.rollout_batch_size
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

sampling_params = {
    "temperature": sampling_temperature,
    "max_tokens": sampling_max_tokens,
    "n": group_size,
    "seed": seed,
    "stop": args.sampling_stop,
    "include_stop_str_in_output": args.include_stop_str_in_output,
}

import os
from dotenv import load_dotenv
load_dotenv()
print(f"HF_HOME: {os.getenv('HF_HOME')}")

import wandb
wandb.login()
wandb_config = {
    "lr": learning_rate,
    "seed": seed,
}
wandb_run = wandb.init(project=wandb_project_name, name=wandb_exp_name, config=wandb_config)

# Seeding
import torch
import random

# TODO: Set random seeds for numpy, torch, ...
random.seed(seed)

# Load dataset
import json

with open(prompt_path) as f:
    prompt_template = f.read()

full_dataset = list()
with open("../data/gsm8k/train.jsonl") as f:
    for line in f:
        row = json.loads(line)
        row["prompt"] = prompt_template.replace("{question}", row["question"])
        row["answer"] = row["answer"].split("####")[-1].strip()
        full_dataset.append(row)
random.shuffle(full_dataset)
train_dataset = full_dataset[:n_train_examples]
valid_dataset = full_dataset[n_train_examples:n_train_examples+n_val_examples]


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
from vllm_utils import VLLMServer

llm_rollout = VLLMServer(
    model_id=model_id,
    gpu=rollout_device,
    seed=seed,
    gpu_memory_utilization=gpu_memory_utilization,
    weight_transfer_backend=weight_transfer_backend,
    enable_lora=use_peft,
    max_lora_rank=lora_r,
    max_loras=1,
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

# Training loop
from grpo_core_implementation import grpo_train_step, track_cuda_memory_and_time
from drgrpo_grader import r1_zero_reward_fn

# NOTE: currently just train for one epoch to avoid overfitting, so I add the following check
assert num_rollout_steps * rollout_batch_size == n_train_examples * group_size
assert (rollout_batch_size / group_size).is_integer()
n_questions_per_rollout = rollout_batch_size // group_size
print(f"Rollout batch size: {rollout_batch_size}; # Questions per rollout: {n_questions_per_rollout}; # Generations per question: {group_size}")

from tqdm import tqdm

for i in tqdm(range(num_rollout_steps), desc="GRPO training steps"):
    time_metrics = {}
    train_rows = train_dataset[i*n_questions_per_rollout:(i+1)*n_questions_per_rollout]
    print(f"Generating rollouts for the following {len(train_rows)} questions (answers): ", [train_row["question"].split()[0] + f" ({train_row['answer']})" for train_row in train_rows])
    # Curate repeated prompts for generating rollouts
    vllm_prompts, prompts, answers = list(), list(), list()
    for train_row in train_rows:
        vllm_prompts.append(train_row["prompt"])
        prompts.extend([train_row["prompt"]] * group_size)
        answers.extend([train_row["answer"]] * group_size)
    assert len(prompts) == len(answers) == rollout_batch_size
    # Generate rollouts
    print(f"Generating {len(prompts)} rollouts with vLLM...")
    with track_cuda_memory_and_time(
        "rollout_full_batch",
        time_metrics=time_metrics,
        track_memory=False,
        track_time=track_step_time,
    ):
        completions = llm_rollout.generate_completions(
            prompts=vllm_prompts,
            sampling_params=sampling_params,  # NOTE: "n": group_size --> so I use vllm_prompts instead of prompts
            batch_size=rollout_batch_size
        )
        responses = [completion.text for completion in completions]
        print(f"Successfully generated {len(responses)} rollouts for {len(prompts)} prompts!")
        assert len(prompts) == len(responses)
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
        **memory_metrics,
        **time_metrics,
    }, step=i)
    # TODO: validation

# Closing
llm_rollout.stop()
if runtime_adapter_dir is not None:
    runtime_adapter_dir.cleanup()

output_dir = f"checkpoints/{wandb_exp_name}-final"
tokenizer.save_pretrained(output_dir)
if use_peft:  # NOTE: currently save the full merged model instead of the adapter only
    merged = llm_policy.merge_and_unload()  # XXX: understand this
    merged.save_pretrained(output_dir, safe_serialization=True)
else:
    llm_policy.save_pretrained(output_dir, safe_serialization=True)

wandb_run.finish()
