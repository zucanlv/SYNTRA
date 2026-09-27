#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"

THEOREMQA_THEOREMS_REASON_EMBED_CONFIG="${PIPELINE_DIR}/config/RQ-2-generalization/reasoning-intensive/config_0712-bright-documents-theoremqa_theorems-reason-embed.yaml"
THEOREMQA_THEOREMS_10K_CONFIG="${PIPELINE_DIR}/config/RQ-2-generalization/reasoning-intensive/config_0712-bright-documents-theoremqa_theorems-10k.yaml"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

run_config() {
  local step_name="$1"
  local config_path="$2"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${step_name}"
  NO_PROXY=127.0.0.1,localhost,0.0.0.0 \
  no_proxy=127.0.0.1,localhost,0.0.0.0 \
  python main.py --config "${config_path}"
}

require_file "${THEOREMQA_THEOREMS_REASON_EMBED_CONFIG}"
require_file "${THEOREMQA_THEOREMS_10K_CONFIG}"

cd "${PIPELINE_DIR}"

run_config "Step 1: bright-documents theoremqa_theorems reason-embed" "${THEOREMQA_THEOREMS_REASON_EMBED_CONFIG}"
run_config "Step 2: bright-documents theoremqa_theorems 10k" "${THEOREMQA_THEOREMS_10K_CONFIG}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
