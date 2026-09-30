#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

seed=42
model_id="Qwen/Qwen3-1.7B-Base"
prompt_name="qwen3_recommended_prompt_math_with_cc"
stop_tokens="}."
lr=5e-5
difficulty_filter="none"
difficulty_lower_bound=0.0
difficulty_upper_bound=1.0
# confidence_recalibration_warmup_steps=5
# confidence_recalibration_history_steps=5

# confidence_estimator_lora_dir="/nfs/brian-wu/stanford-cs336-assignment5-alignment/outputs/confidence_training/gsm8k_offline_qwen3_1.7b_base/gsm8k__Qwen3-1.7B-Base__lora__token-nll__dataset-O-10__lr-1e-4__bs-16__seed-42__old/checkpoints/epoch_000002/model"
dataset="triviaqa"
suffix="__triviaqa-implementation-only-accept-boxed"
base_exp_name="${model_id}__${dataset}__${prompt_name}__lr-${lr}__max-tokens-256__lora-r-16-a-32-fp16__difficulty-filter-by-${difficulty_filter}${suffix}__seed-${seed}"

uv run training_script_grpo.py \
  --wandb-project-name "Efficient_GRPO_GSM8K" \
  --wandb-exp-name "${base_exp_name}" \
  --model-id "${model_id}" \
  --policy-device 0 \
  --rollout-device 1 \
  --gpu-memory-utilization 0.9 \
  --dataset-folder "../data/${dataset}" \
  --prompt-path "prompts/${prompt_name}.prompt" \
  --no-add-cc-sft-loss \
  --n-train-examples 6400 \
  --n-val-examples 1024 \
  --num-rollout-steps 200 \
  --learning-rate "${lr}" \
  --vllm-max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --train-batch-size 256 \
  --valid-batch-size 1024 \
  --difficulty-filter "${difficulty_filter}" \
  --difficulty-lower-bound "${difficulty_lower_bound}" \
  --difficulty-upper-bound "${difficulty_upper_bound}" \
  --max-candidate-groups-multiplier 200 \
  --group-size 8 \
  --gradient-accumulation-steps 32 \
  --sampling-temperature 1.0 \
  --sampling-max-tokens 256 \
  --sampling-stop "${stop_tokens}" \
  --include-stop-str-in-output \
  --max-grad-norm 1.0 \
  --adamw-betas 0.9 0.95 \
  --weight-decay 0.0 \
  --seed "${seed}" \
  --track-policy-memory \
  --use-peft \
  --peft-method lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --lora-adapter-name policy \
  --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --no-autocast-adapter-dtype \
  --save-only-adapter \
  "$@"
