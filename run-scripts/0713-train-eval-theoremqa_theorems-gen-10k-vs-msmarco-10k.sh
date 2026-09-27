#!/usr/bin/env bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
TRAIN_PROJECT="/data/share/project/tr/data_synthesis_agent"
TASK="theoremqa_theorems"
BUILD_SCRIPT="${PROJECT_ROOT}/run-scripts/0713-build-bright-theoremqa_theorems-gen-10k-training-data.sh"
TRAIN_SCRIPT="${TRAIN_PROJECT}/code/scripts/7-11/RQ2/train-qwen3_0.6b.sh"
EVAL_SCRIPT="${TRAIN_PROJECT}/code/scripts/7-11/RQ2/eval_bright_singal_task.sh"
PATHS_FILE="/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task/latest-theoremqa_theorems-gen-10k-training-data.paths"
MSMARCO_MODEL="${TRAIN_PROJECT}/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1"
GEN_MODEL="${TRAIN_PROJECT}/model_pth/7-13/RQ2/${TASK}/train-qwen3_0.6b-gen-10k"
LOG_DIR="${TRAIN_PROJECT}/logs/7-13/RQ2/${TASK}"
EVAL_ROOT="${TRAIN_PROJECT}/eval_output/7-13-bright/RQ2/${TASK}"

mkdir -p "${LOG_DIR}" "${EVAL_ROOT}"
bash "${BUILD_SCRIPT}" 2>&1 | tee "${LOG_DIR}/build-gen-10k-data.log"
mapfile -t data_paths < "${PATHS_FILE}"
if [[ ${#data_paths[@]} -ne 2 ]]; then
  echo "Expected two generated training-data paths in ${PATHS_FILE}" >&2
  exit 1
fi

bash "${TRAIN_SCRIPT}" "${GEN_MODEL}" "${data_paths[*]}" 2>&1 | tee "${LOG_DIR}/train-qwen3_0.6b-gen-10k.log"
bash "${EVAL_SCRIPT}" "${MSMARCO_MODEL}" "${EVAL_ROOT}/train-qwen3_0.6b-msmarco-10k" "${TASK}" 2>&1 | tee "${LOG_DIR}/eval-qwen3_0.6b-msmarco-10k.log"
bash "${EVAL_SCRIPT}" "${GEN_MODEL}" "${EVAL_ROOT}/train-qwen3_0.6b-gen-10k" "${TASK}" 2>&1 | tee "${LOG_DIR}/eval-qwen3_0.6b-gen-10k.log"

