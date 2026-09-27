#!/usr/bin/env bash

set -euo pipefail

TRAIN_SCRIPT="/data/share/project/tr/data_synthesis_agent/code/scripts/7-20/RQ2/train-qwen3_0.6b.sh"
BEIR_EVAL_SCRIPT="/data/share/project/tr/data_synthesis_agent/code/scripts/7-20/RQ2/eval_beir.sh"

DATA_ROOT="/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/msmarco"
MODEL_ROOT="/data/share/project/tr/data_synthesis_agent/model_pth/7-20/RQ2/msmarco"
LOG_ROOT="/data/share/project/tr/data_synthesis_agent/logs/7-20/RQ2/msmarco"
EVAL_ROOT="/data/share/project/tr/data_synthesis_agent/eval_output/7-20-beir/RQ2/msmarco"

TAG="deepseek-v4-flash-20260720_005250"
TASK="msmarco"
TRAIN_FILES=(
  "${DATA_ROOT}/${TAG}.jsonl"
  "${DATA_ROOT}/${TAG}_exclude_other_pos_save_non_pos.jsonl"
)

MODEL_DIR="${MODEL_ROOT}/train-qwen3_0.6b-merged-${TAG}"
LOG_PATH="${LOG_ROOT}/train-qwen3_0.6b-merged-${TAG}.log"
EVAL_DIR="${EVAL_ROOT}/train-qwen3_0.6b-merged-${TAG}"

required_files=(
  "${TRAIN_SCRIPT}"
  "${BEIR_EVAL_SCRIPT}"
  "${TRAIN_FILES[@]}"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done

printf -v train_data '%s ' "${TRAIN_FILES[@]}"
train_data="${train_data% }"

mkdir -p "${LOG_ROOT}"

echo "============================================================"
echo "Task       : ${TASK}"
echo "Tag        : ${TAG}"
echo "Train data : ${train_data}"
echo "Model dir  : ${MODEL_DIR}"
echo "Eval dir   : ${EVAL_DIR}"
echo "============================================================"

bash "${TRAIN_SCRIPT}" "${MODEL_DIR}" "${train_data}" 2>&1 | tee "${LOG_PATH}"
bash "${BEIR_EVAL_SCRIPT}" "${MODEL_DIR}" "${EVAL_DIR}" "${TASK}"
