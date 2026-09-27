#!/usr/bin/env bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
TRAIN_PROJECT="/data/share/project/tr/data_synthesis_agent"
TASK="theoremqa_theorems"

BUILD_SCRIPT="${PROJECT_ROOT}/run-scripts/0712-build-bright-theoremqa_theorems-training-data.sh"
TRAIN_SCRIPT="${TRAIN_PROJECT}/code/scripts/7-11/RQ2/train-qwen3_0.6b.sh"
EVAL_SCRIPT="${TRAIN_PROJECT}/code/scripts/7-11/RQ2/eval_bright_singal_task.sh"

DATA_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task"
DATA_STEM="bright-documents-theoremqa_theorems-20260712-reason-embed-query"
TRAIN_DATA="${DATA_DIR}/${DATA_STEM}.jsonl ${DATA_DIR}/${DATA_STEM}_exclude_other_pos_save_non_pos.jsonl"

MSMARCO_MODEL="${TRAIN_PROJECT}/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1"
REASON_MODEL="${TRAIN_PROJECT}/model_pth/7-12/RQ2/${TASK}/train-qwen3_0.6b-reason-embed"

LOG_DIR="${TRAIN_PROJECT}/logs/7-12/RQ2/${TASK}"
EVAL_ROOT="${TRAIN_PROJECT}/eval_output/7-12-bright/RQ2/${TASK}"
MSMARCO_EVAL_DIR="${EVAL_ROOT}/train-qwen3_0.6b-msmarco-10k"
REASON_EVAL_DIR="${EVAL_ROOT}/train-qwen3_0.6b-reason-embed"

for required_file in "${BUILD_SCRIPT}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${MSMARCO_MODEL}" ]]; then
  echo "Missing MSMARCO-10k model: ${MSMARCO_MODEL}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${EVAL_ROOT}"

echo "[1/4] Build theoremqa_theorems reason-embed training data"
bash "${BUILD_SCRIPT}"

echo "[2/4] Train theoremqa_theorems reason-embed model"
bash "${TRAIN_SCRIPT}" "${REASON_MODEL}" "${TRAIN_DATA}" \
  2>&1 | tee "${LOG_DIR}/train-qwen3_0.6b-reason-embed.log"

echo "[3/4] Evaluate MSMARCO-10k model on theoremqa_theorems"
bash "${EVAL_SCRIPT}" "${MSMARCO_MODEL}" "${MSMARCO_EVAL_DIR}" "${TASK}" \
  2>&1 | tee "${LOG_DIR}/eval-qwen3_0.6b-msmarco-10k.log"

echo "[4/4] Evaluate reason-embed model on theoremqa_theorems"
bash "${EVAL_SCRIPT}" "${REASON_MODEL}" "${REASON_EVAL_DIR}" "${TASK}" \
  2>&1 | tee "${LOG_DIR}/eval-qwen3_0.6b-reason-embed.log"

echo "MSMARCO-10k result: ${MSMARCO_EVAL_DIR}/eval_results_examples.md"
echo "Reason-embed result: ${REASON_EVAL_DIR}/eval_results_examples.md"

