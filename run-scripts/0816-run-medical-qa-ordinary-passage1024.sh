#!/usr/bin/env bash

set -euo pipefail

if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/ustc/public_envs/DSA
fi

export WANDB_MODE=disabled
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/share/project/shared_datasets/.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/share/project/shared_datasets/mteb}"
export MTEB_CACHE="${MTEB_CACHE:-/data/share/project/shared_datasets/mteb}"

DATA_DIR="${DATA_DIR:-/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medical_qa}"
MODEL_ROOT="${MODEL_ROOT:-/data/share/project/tr/data_synthesis_agent/model_pth/8-16/RQ2/MedicalRetrieval-ordinary-passage1024}"
EVAL_ROOT="${EVAL_ROOT:-/data/share/project/tr/data_synthesis_agent/eval_output/8-16-mteb/RQ2/MedicalRetrieval-ordinary-passage1024}"
LOG_ROOT="${LOG_ROOT:-/data/share/project/tr/data_synthesis_agent/logs/8-16/RQ2/MedicalRetrieval-ordinary-passage1024}"

MODEL_PATH="${MODEL_PATH:-/data/share/project/shared_models/Qwen3-0.6B-auto-eos}"
MTEB_MAIN="${MTEB_MAIN:-/data/share/project/tr/mmteb/code/train/qwen3-8b/code/main.py}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-/data/share/project/public_envs/FlagEmbedding/examples/finetune/ds_stage1.json}"
SENTENCE_TRANSFORMER_TEMPLATE="${SENTENCE_TRANSFORMER_TEMPLATE:-/data/share/project/shared_models/Qwen3-Embedding-0.6B}"
EVAL_PYTHON="${EVAL_PYTHON:-/data/share/project/public_envs/embedder_eval_latest/bin/python}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-4,5}"
NUM_GPUS="${NUM_GPUS:-6}"
MASTER_PORT="${MASTER_PORT:-29517}"
PASSAGE_MAX_LEN=1024
QUERY_MAX_LEN=512

TRAIN_1Q=(
  "${DATA_DIR}/medical_qa-all-1q-20260806_exclude_other_pos_save_non_pos.jsonl"
  "${DATA_DIR}/medical_qa-all-1q-20260806.jsonl"
)
TRAIN_2Q=(
  "${DATA_DIR}/medical_qa-all-2q-query1-20260811_exclude_other_pos_save_non_pos.jsonl"
  "${DATA_DIR}/medical_qa-all-2q-query1-20260811.jsonl"
  "${DATA_DIR}/medical_qa-all-2q-query2-20260811_exclude_other_pos_save_non_pos.jsonl"
  "${DATA_DIR}/medical_qa-all-2q-query2-20260811.jsonl"
)
TRAIN_4Q=(
  "${DATA_DIR}/medical_qa-all-4q-query1-20260812_exclude_other_pos_save_non_pos.jsonl"
  "${DATA_DIR}/medical_qa-all-4q-query1-20260812.jsonl"
  "${DATA_DIR}/medical_qa-all-4q-query2-20260812_exclude_other_pos_save_non_pos.jsonl"
  "${DATA_DIR}/medical_qa-all-4q-query2-20260812.jsonl"
  "${DATA_DIR}/medical_qa-all-4q-query3-20260812_exclude_other_pos_save_non_pos.jsonl"
  "${DATA_DIR}/medical_qa-all-4q-query3-20260812.jsonl"
  "${DATA_DIR}/medical_qa-all-4q-query4-20260812_exclude_other_pos_save_non_pos.jsonl"
  "${DATA_DIR}/medical_qa-all-4q-query4-20260812.jsonl"
)

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

print_cmd() {
  printf ' %s' "$@"
  printf '\n'
}

preflight() {
  [[ -d "${MODEL_PATH}" ]] || {
    echo "Missing base model directory: ${MODEL_PATH}" >&2
    exit 1
  }
  require_file "${MTEB_MAIN}"
  require_file "${DEEPSPEED_CONFIG}"
  require_file "${EVAL_PYTHON}"
  require_file "${SENTENCE_TRANSFORMER_TEMPLATE}/1_Pooling/config.json"
  require_file "${SENTENCE_TRANSFORMER_TEMPLATE}/config_sentence_transformers.json"
  require_file "${SENTENCE_TRANSFORMER_TEMPLATE}/modules.json"
  require_file "${SENTENCE_TRANSFORMER_TEMPLATE}/sentence_bert_config.json"
  for data_file in "${TRAIN_1Q[@]}" "${TRAIN_2Q[@]}" "${TRAIN_4Q[@]}"; do
    require_file "${data_file}"
  done
}

copy_sentence_transformer_files() {
  local merged_model="$1"
  mkdir -p "${merged_model}/1_Pooling"
  cp "${SENTENCE_TRANSFORMER_TEMPLATE}/1_Pooling/config.json" "${merged_model}/1_Pooling/config.json"
  cp "${SENTENCE_TRANSFORMER_TEMPLATE}/config_sentence_transformers.json" "${merged_model}/config_sentence_transformers.json"
  cp "${SENTENCE_TRANSFORMER_TEMPLATE}/modules.json" "${merged_model}/modules.json"
  cp "${SENTENCE_TRANSFORMER_TEMPLATE}/sentence_bert_config.json" "${merged_model}/sentence_bert_config.json"
}

run_train() {
  local experiment="$1"
  shift
  local model_dir="${MODEL_ROOT}/${experiment}"
  local train_log="${LOG_ROOT}/${experiment}-train.log"
  local train_cmd=(
    torchrun --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}"
    -m FlagEmbedding.finetune.embedder.decoder_only.base
    --model_name_or_path "${MODEL_PATH}"
    --cache_dir "${HF_HUB_CACHE}"
    --use_lora True
    --lora_rank 32
    --lora_alpha 64
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj
    --save_merged_lora_model True
    --use_mrl False
    --trust_remote_code True
    --train_data "$@"
    --cache_path "${HF_HUB_CACHE}"
    --train_group_size 8
    --query_max_len "${QUERY_MAX_LEN}"
    --passage_max_len "${PASSAGE_MAX_LEN}"
    --pad_to_multiple_of 8
    --query_instruction_format 'Instruct: {}\nQuery: {}'
    --knowledge_distillation True
    --same_dataset_within_batch True
    --small_threshold 0
    --drop_threshold 0
    --output_dir "${model_dir}"
    --overwrite_output_dir
    --learning_rate 1e-4
    --bf16
    --num_train_epochs 1
    --per_device_train_batch_size 16
    --dataloader_drop_last True
    --warmup_ratio 0.1
    --gradient_checkpointing
    --deepspeed "${DEEPSPEED_CONFIG}"
    --logging_steps 1
    --save_steps 1000
    --negatives_cross_device
    --temperature 0.02
    --sentence_pooling_method last_token
    --normalize_embeddings True
    --kd_loss_type kl_div
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN TRAIN experiment=${experiment}"
    printf 'CUDA_VISIBLE_DEVICES=%s' "${TRAIN_CUDA_VISIBLE_DEVICES}"
    print_cmd "${train_cmd[@]}"
    return 0
  fi

  mkdir -p "${model_dir}" "${LOG_ROOT}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] TRAIN experiment=${experiment}"
  CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" "${train_cmd[@]}" 2>&1 | tee "${train_log}"
}

run_eval() {
  local experiment="$1"
  local merged_model="${MODEL_ROOT}/${experiment}/merged_model"
  local output_dir="${EVAL_ROOT}/${experiment}"
  local eval_log="${LOG_ROOT}/${experiment}-eval.log"
  local eval_cmd=(
    "${EVAL_PYTHON}" "${MTEB_MAIN}"
    --benchmark_name MedicalRetrieval
    --results_output_folder "${output_dir}"
    --model_name_or_path "${merged_model}"
    --use_fp16 False
    --use_bf16 True
    --prompt_template 'Instruct: {}\nQuery: '
    --assert_prompts_exist False
    --normalize_embeddings True
    --batch_size 4
    --max_length "${PASSAGE_MAX_LEN}"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN EVAL experiment=${experiment}"
    printf 'CUDA_VISIBLE_DEVICES=%s' "${EVAL_CUDA_VISIBLE_DEVICES}"
    print_cmd "${eval_cmd[@]}"
    return 0
  fi

  [[ -d "${merged_model}" ]] || {
    echo "Missing merged model: ${merged_model}" >&2
    exit 1
  }
  copy_sentence_transformer_files "${merged_model}"
  mkdir -p "${output_dir}" "${LOG_ROOT}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EVAL experiment=${experiment}"
  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" "${eval_cmd[@]}" 2>&1 | tee "${eval_log}"
}

run_experiment() {
  local experiment="$1"
  shift
  if [[ "${SKIP_EXISTING_TRAIN:-0}" == "1" && -d "${MODEL_ROOT}/${experiment}/merged_model" ]]; then
    echo "SKIP TRAIN experiment=${experiment}: merged_model already exists"
  else
    run_train "${experiment}" "$@"
  fi
  run_eval "${experiment}"
}

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  preflight
fi

run_experiment 1q "${TRAIN_1Q[@]}"
run_experiment 2q "${TRAIN_2Q[@]}"
run_experiment 4q "${TRAIN_4Q[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed MedicalRetrieval ordinary-data passage-1024 experiments"
