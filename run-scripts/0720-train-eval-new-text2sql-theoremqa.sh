#!/usr/bin/env bash

set -euo pipefail

TRAIN_SCRIPT="/data/share/project/tr/data_synthesis_agent/code/scripts/7-20/RQ2/train-qwen3_0.6b.sh"
COIR_EVAL_SCRIPT="/data/share/project/tr/data_synthesis_agent/code/scripts/7-20/RQ2/eval_coir_singal_task.sh"
BRIGHT_EVAL_SCRIPT="/data/share/project/tr/data_synthesis_agent/code/scripts/7-11/RQ2/eval_bright_singal_task.sh"

DATA_ROOT="/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs"
MODEL_ROOT="/data/share/project/tr/data_synthesis_agent/model_pth/7-20/RQ2"
LOG_ROOT="/data/share/project/tr/data_synthesis_agent/logs/7-20/RQ2"
COIR_EVAL_ROOT="/data/share/project/tr/data_synthesis_agent/eval_output/7-20-coir/RQ2"
BRIGHT_EVAL_ROOT="/data/share/project/tr/data_synthesis_agent/eval_output/7-20-bright/RQ2"

TEXT2SQL_TAG="deepseek-v4-flash-20260720_010219"
TEXT2SQL_TASK="synthetic-text2sql"
TEXT2SQL_DATA=(
  "${DATA_ROOT}/synthetic-text2sql/${TEXT2SQL_TAG}.jsonl"
  "${DATA_ROOT}/synthetic-text2sql/${TEXT2SQL_TAG}_exclude_other_pos_save_non_pos.jsonl"
)

THEOREMQA_TAG="deepseek-v4-flash-20260720_010413"
THEOREMQA_TASK="theoremqa_theorems"
THEOREMQA_DATA=(
  "${DATA_ROOT}/theoremqa-theorem/${THEOREMQA_TAG}.jsonl"
  "${DATA_ROOT}/theoremqa-theorem/${THEOREMQA_TAG}_exclude_other_pos_save_non_pos.jsonl"
)

required_files=(
  "${TRAIN_SCRIPT}"
  "${COIR_EVAL_SCRIPT}"
  "${BRIGHT_EVAL_SCRIPT}"
  "${TEXT2SQL_DATA[@]}"
  "${THEOREMQA_DATA[@]}"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done

run_train_and_eval() {
  local task="$1"
  local tag="$2"
  local eval_script="$3"
  local eval_root="$4"
  shift 4
  local train_files=("$@")
  local train_data
  local model_dir="${MODEL_ROOT}/${task}/train-qwen3_0.6b-merged-${tag}"
  local log_dir="${LOG_ROOT}/${task}"
  local log_path="${log_dir}/train-qwen3_0.6b-merged-${tag}.log"
  local eval_dir="${eval_root}/${task}/train-qwen3_0.6b-merged-${tag}"

  printf -v train_data '%s ' "${train_files[@]}"
  train_data="${train_data% }"

  mkdir -p "${log_dir}"

  echo "============================================================"
  echo "Task       : ${task}"
  echo "Tag        : ${tag}"
  echo "Train data : ${train_data}"
  echo "Model dir  : ${model_dir}"
  echo "Eval dir   : ${eval_dir}"
  echo "============================================================"

  bash "${TRAIN_SCRIPT}" "${model_dir}" "${train_data}" 2>&1 | tee "${log_path}"
  bash "${eval_script}" "${model_dir}" "${eval_dir}" "${task}"
}

# CoIR: train on the two synthetic-text2sql JSONL files, then evaluate only
# the synthetic-text2sql task with the CoIR evaluator.
run_train_and_eval \
  "${TEXT2SQL_TASK}" \
  "${TEXT2SQL_TAG}" \
  "${COIR_EVAL_SCRIPT}" \
  "${COIR_EVAL_ROOT}" \
  "${TEXT2SQL_DATA[@]}"

# BRIGHT: train on the two theoremqa JSONL files, then evaluate the
# theoremqa_theorems examples split with BRIGHT's task-specific instruction.
run_train_and_eval \
  "${THEOREMQA_TASK}" \
  "${THEOREMQA_TAG}" \
  "${BRIGHT_EVAL_SCRIPT}" \
  "${BRIGHT_EVAL_ROOT}" \
  "${THEOREMQA_DATA[@]}"
