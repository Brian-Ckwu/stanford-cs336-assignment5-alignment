#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

model_id="Qwen/Qwen3-8B"
lr=1e-5
# _lora-r-16-a-32-dropout-0-bf16
uv run training_script_grpo_on_cc.py \
  --wandb-project-name "GRPO_CC_TriviaQA" \
  --wandb-exp-name "${model_id}_lr-${lr}_lora-r-16-a-32-dropout-0-bf16_h100" \
  --model-id "${model_id}" \
  --dataset_path "../data/capability_calibration/triviaqa-train__Qwen3-8B-non-thinking/grpo_dataset.jsonl" \
  --policy-device 0 \
  --rollout-device 1 \
  --gpu-memory-utilization 0.9 \
  --prompt-path "prompts/verbalized_cc_5-shots.prompt" \
  --n-train-examples 12800 \
  --n-val-examples 1024 \
  --num-rollout-steps 400 \
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
