#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <output_dir> <synthetic_train_data.jsonl>" >&2
  exit 2
fi

export PATH="/data/share/project/public_envs/embedder_train_eval/bin:$PATH"
export WANDB_MODE=disabled
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/share/project/shared_datasets/.cache}"

MODEL_PATH="${QWEN3_8B_MODEL_PATH:-/data/share/project/shared_models/Qwen3-Embedding-8B}"
OUTPUT_DIR="$1"
TRAIN_DATA="$2"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#GPU_IDS[@]}"

[[ -d "$MODEL_PATH" ]] || { echo "Missing Qwen3-Embedding-8B model: $MODEL_PATH" >&2; exit 2; }
[[ -f "$TRAIN_DATA" ]] || { echo "Missing training data: $TRAIN_DATA" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"

echo "Qwen3-Embedding-8B training: model=$MODEL_PATH data=$TRAIN_DATA gpus=$CUDA_VISIBLE_DEVICES"

torchrun --nproc_per_node "$NUM_GPUS" --master_port "${MASTER_PORT:-29531}" \
  -m FlagEmbedding.finetune.embedder.decoder_only.base \
  --model_name_or_path "$MODEL_PATH" --cache_dir "$HF_HUB_CACHE" \
  --use_lora True --lora_rank 32 --lora_alpha 64 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
  --save_merged_lora_model True --use_mrl False --trust_remote_code True \
  --train_data "$TRAIN_DATA" --cache_path "$HF_HUB_CACHE" \
  --train_group_size 8 --query_max_len 512 --passage_max_len 512 --pad_to_multiple_of 8 \
  --query_instruction_format 'Instruct: {}\nQuery: {}' --knowledge_distillation True \
  --same_dataset_within_batch True --small_threshold 0 --drop_threshold 0 \
  --output_dir "$OUTPUT_DIR" --overwrite_output_dir \
  --learning_rate "${LEARNING_RATE:-1e-4}" --bf16 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-16}" \
  --dataloader_drop_last True --warmup_ratio 0.1 --gradient_checkpointing \
  --deepspeed /data/share/project/public_envs/FlagEmbedding/examples/finetune/ds_stage1.json \
  --logging_steps 1 --save_steps 1000 --negatives_cross_device --temperature 0.02 \
  --sentence_pooling_method last_token --normalize_embeddings True --kd_loss_type kl_div
