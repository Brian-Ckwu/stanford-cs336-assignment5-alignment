#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

model_id="Qwen/Qwen3-8B"
learning_rate="3e-5"
optimization_mode="bce_with_logits"
train_examples=6400

uv run training_sft_on_cc.py \
  --wandb-project-name "SFT_CC_TriviaQA" \
  --wandb-exp-name "${model_id}_lr-${learning_rate}_lora-r-16-a-32-dropout-0-bf16_train-${train_examples}_default-chat-question-only_${optimization_mode}" \
  --model-id "${model_id}" \
  --dataset_path "../data/capability_calibration/triviaqa-train__Qwen3-8B-non-thinking/grpo_dataset.jsonl" \
  --device 0 \
  --prompt-path "prompts/question_only.prompt" \
  --use-chat-template \
  --n-train-examples "${train_examples}" \
  --n-val-examples 1024 \
  --num-train-epochs 1 \
  --learning-rate "${learning_rate}" \
  --optimization-mode "${optimization_mode}" \
  --train-batch-size 32 \
  --validation-batch-size 32 \
  --gradient-accumulation-steps 32 \
  --sequence-max-tokens 1024 \
  --validation-interval 10 \
  --max-grad-norm 1.0 \
  --adamw-betas 0.9 0.95 \
  --weight-decay 0.0 \
  --seed 42 \
  --output-dir "checkpoints" \
  --dtype bfloat16 \
  --track-policy-memory \
  --track-step-time \
  --use-peft \
  --peft-method lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --no-autocast-adapter-dtype
