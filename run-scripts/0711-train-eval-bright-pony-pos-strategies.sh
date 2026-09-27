#!/usr/bin/env bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

TRAIN_SCRIPT="/data/share/project/tr/data_synthesis_agent/code/scripts/7-11/RQ2/train-qwen3_0.6b.sh"
EVAL_SCRIPT="/data/share/project/tr/data_synthesis_agent/code/scripts/7-11/RQ2/eval_bright_singal_task.sh"
DATA_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task"
MODEL_DIR="/data/share/project/tr/data_synthesis_agent/model_pth/7-11/RQ2/pony"
LOG_DIR="/data/share/project/tr/data_synthesis_agent/logs/7-11/RQ2/pony"
EVAL_DIR="/data/share/project/tr/data_synthesis_agent/eval_output/7-11-bright/RQ2/pony"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

run_experiment() {
  local strategy="$1"
  local source="$2"
  local model_name="train-qwen3_0.6b-${strategy}-${source}"
  local data_prefix="bright-documents-pony-20260710-${source}-query-${strategy}"
  local train_data="${DATA_DIR}/${data_prefix}.jsonl ${DATA_DIR}/${data_prefix}_exclude_other_pos_save_non_pos.jsonl"
  local model_path="${MODEL_DIR}/${model_name}"
  local log_path="${LOG_DIR}/${model_name}.log"
  local eval_path="${EVAL_DIR}/${model_name}"

  require_file "${DATA_DIR}/${data_prefix}.jsonl"
  require_file "${DATA_DIR}/${data_prefix}_exclude_other_pos_save_non_pos.jsonl"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training ${model_name}"
  bash "${TRAIN_SCRIPT}" "${model_path}" "${train_data}" 2>&1 | tee "${log_path}"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluating ${model_name}"
  bash "${EVAL_SCRIPT}" "${model_path}" "${eval_path}" pony 2>&1 | tee -a "${log_path}"
}

require_file "${TRAIN_SCRIPT}"
require_file "${EVAL_SCRIPT}"
mkdir -p "${MODEL_DIR}" "${LOG_DIR}" "${EVAL_DIR}"

run_experiment "score3-only" "reason-embed"
run_experiment "score3-only" "gen"
run_experiment "score3-fallback-score2" "reason-embed"
run_experiment "score3-fallback-score2" "gen"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All Pony strategy experiments completed."
