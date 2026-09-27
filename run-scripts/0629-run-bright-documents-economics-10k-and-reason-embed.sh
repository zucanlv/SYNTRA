#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"

ECONOMICS_10K_CONFIG="${PIPELINE_DIR}/config/RQ-2-generalization/reasoning-intensive/config_0629-bright-documents-economics-10k.yaml"
REASON_EMBED_CONFIG="${PIPELINE_DIR}/config/RQ-2-generalization/reasoning-intensive/config_0629-bright-documents-economics-reason-embed.yaml"

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

require_file "${ECONOMICS_10K_CONFIG}"
require_file "${REASON_EMBED_CONFIG}"

cd "${PIPELINE_DIR}"

run_config "Step 1: bright-documents economics 10k" "${ECONOMICS_10K_CONFIG}"
run_config "Step 2: bright-documents economics reason-embed" "${REASON_EMBED_CONFIG}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
