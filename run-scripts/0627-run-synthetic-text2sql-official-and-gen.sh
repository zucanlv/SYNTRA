#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"

OFFICIAL_CONFIG="${PIPELINE_DIR}/config/RQ-2-generalization/code-retrieval/config-0627-synthetic-text2sql-official.yaml"
GEN_CONFIG="${PIPELINE_DIR}/config/RQ-2-generalization/code-retrieval/config-0627-synthetic-text2sql-official-doc-gen-qeury.yaml"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_file "${OFFICIAL_CONFIG}"
require_file "${GEN_CONFIG}"

cd "${PIPELINE_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1: synthetic-text2sql official"
NO_PROXY=127.0.0.1,localhost,0.0.0.0 \
no_proxy=127.0.0.1,localhost,0.0.0.0 \
python main.py --config "${OFFICIAL_CONFIG}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2: synthetic-text2sql official-doc generated-query"
NO_PROXY=127.0.0.1,localhost,0.0.0.0 \
no_proxy=127.0.0.1,localhost,0.0.0.0 \
python main.py --config "${GEN_CONFIG}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
