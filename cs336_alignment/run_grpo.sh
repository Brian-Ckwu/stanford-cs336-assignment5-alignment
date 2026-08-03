#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

lr=1e-4

uv run training_script_grpo.py \
  --wandb-project-name "OLMo-2-0425-1B_GRPO_GSM8K" \
  --wandb-exp-name "r1-zero-prompt_lr-${lr}_max-tokens-256_lora-r-16-a-32-dropout-0-fp32" \
  --model-id "allenai/OLMo-2-0425-1B" \
  --policy-device 2 \
  --rollout-device 3 \
  --gpu-memory-utilization 0.9 \
  --prompt-path "prompts/r1_zero.prompt" \
  --n-train-examples 6400 \
  --n-val-examples 1024 \
  --num-rollout-steps 200 \
  --learning-rate "${lr}" \
  --rollout-batch-size 256 \
  --train-batch-size 256 \
  --group-size 8 \
  --gradient-accumulation-steps 32 \
  --sampling-temperature 1.0 \
  --sampling-max-tokens 256 \
  --sampling-stop "</answer>" \
  --include-stop-str-in-output \
  --max-grad-norm 1.0 \
  --adamw-betas 0.9 0.95 \
  --weight-decay 0.0 \
  --seed 42 \
  --track-policy-memory \
  --use-peft \
  --peft-method lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --lora-adapter-name policy \
  --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --autocast-adapter-dtype
