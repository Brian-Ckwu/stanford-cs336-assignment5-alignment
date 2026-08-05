#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

model_id="Qwen/Qwen3-1.7B"
lr=3e-5
# _lora-r-16-a-32-dropout-0-bf16
uv run training_script_grpo_on_cc.py \
  --wandb-project-name "GRPO_CC_TriviaQA" \
  --wandb-exp-name "${model_id}_lr-${lr}_lora-r-16-a-32-dropout-0-bf16_temp-0.6_ntrain-73600" \
  --model-id "${model_id}" \
  --policy-device 2 \
  --rollout-device 3 \
  --gpu-memory-utilization 0.9 \
  --prompt-path "prompts/verbalized_cc.prompt" \
  --n-train-examples 73600 \
  --n-val-examples 2048 \
  --num-rollout-steps 2300 \
  --learning-rate "${lr}" \
  --rollout-batch-size 256 \
  --train-batch-size 256 \
  --group-size 8 \
  --gradient-accumulation-steps 32 \
  --sampling-temperature 0.6 \
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
  --no-autocast-adapter-dtype
