wandb_project_name = "OLMo-2-0425-1B_GRPO_GSM8K"
wandb_exp_name = "r1-zero-prompt_default-hparams_max-tokens-256_dgx-spark"

model_id = "allenai/OLMo-2-0425-1B"
policy_device = 0
rollout_device = 0
gpu_memory_utilization = 0.5
weight_transfer_backend = "ipc" if policy_device == rollout_device else "nccl"
prompt_path = "prompts/r1_zero.prompt"

n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = 1e-5
rollout_batch_size = train_batch_size = 256
group_size = 8
gradient_accumulation_steps = 32
sampling_temperature = 1.0
sampling_max_tokens = 256
max_grad_norm = 1.0
adamw_betas = (0.9, 0.95)
weight_decay = 0.0
seed = 42

sampling_params = {
    "temperature": sampling_temperature,
    "max_tokens": sampling_max_tokens,
    "n": group_size,
    "seed": seed,
    "stop": "</answer>",
    "include_stop_str_in_output": True
}

import os
from dotenv import load_dotenv
load_dotenv()
print(f"HF_HOME: {os.getenv("HF_HOME")}")

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
        row["prompt"] = prompt_template.format(question=row["question"])
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
optimizer = torch.optim.AdamW(
    llm_policy.parameters(), lr=learning_rate, betas=adamw_betas, weight_decay=weight_decay
)

# B: rollout model
from vllm_utils import VLLMServer

llm_rollout = VLLMServer(
    model_id=model_id,
    gpu=rollout_device,
    seed=seed,
    gpu_memory_utilization=gpu_memory_utilization,
    weight_transfer_backend=weight_transfer_backend,
)
print(f"Starting the rollout model (vLLM service)...")
llm_rollout.start()
llm_rollout.init_weight_sync(policy_device=llm_policy_device)  # NOTE: Create the communication channel between two llms

# Training loop
from grpo_core_implementation import grpo_train_step
from drgrpo_grader import r1_zero_reward_fn

# NOTE: currently just train for one epoch to avoid overfitting, so I add the following check
assert num_rollout_steps * rollout_batch_size == n_train_examples * group_size
assert (rollout_batch_size / group_size).is_integer()
n_questions_per_rollout = rollout_batch_size // group_size
print(f"Rollout batch size: {rollout_batch_size}; # Questions per rollout: {n_questions_per_rollout}; # Generations per question: {group_size}")

from tqdm import tqdm

for i in tqdm(range(num_rollout_steps), desc="GRPO training steps"):
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
    )
    wandb_run.log(data={
        "train/loss": train_step_loss,
        **{f"train/{key}": value for key, value in train_step_metadata.items()}
    }, step=i)
    # Sync weights
    print("Syncing weights of the rollout LLM to be the same with the updated policy LLM...")
    llm_rollout.sync_policy_weights(policy=llm_policy)
    print("Syncing done!")
    # TODO: validation

# Closing
llm_rollout.stop()

output_dir = f"checkpoints/{wandb_exp_name}-final"
llm_policy.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

wandb_run.finish()
