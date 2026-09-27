#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PREPROCESS_DIR="${PROJECT_ROOT}/Main_Pipeline/corpus_preprocess"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

RESULT_DIR="${PROJECT_ROOT}/Main_Pipeline/Results/msmarco_20260620_175300"
ANNOTATED_FILE="${RESULT_DIR}/Annotated_Main_20260620_175438.json"
DIVERSE_QUERY_FILE="${RESULT_DIR}/DiverseQuery_FromFile_20260620_175300.json"
OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0618-ori"

TRAINING_DATA_A="${OUTPUT_DIR}/msmarco-ori-13804-0620.jsonl"
TRAINING_DATA_B="${OUTPUT_DIR}/msmarco-ori-13804-0620_exclude_other_pos_save_non_pos.jsonl"
PROMPT="Given a web search query, retrieve relevant passages that answer the query."

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

for required_file in \
  "${ANNO_TO_MSMARCO}" \
  "${EXCLUDE_OTHER_POS}" \
  "${ANNOTATED_FILE}" \
  "${DIVERSE_QUERY_FILE}"; do
  require_file "${required_file}"
done

mkdir -p "${OUTPUT_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1: building training data A"
python "${ANNO_TO_MSMARCO}" \
  --input "${ANNOTATED_FILE}" \
  --output "${TRAINING_DATA_A}" \
  --prompt "${PROMPT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2: building training data B"
python "${EXCLUDE_OTHER_POS}" \
  --input-jsonl "${TRAINING_DATA_A}" \
  --diverse-query "${DIVERSE_QUERY_FILE}" \
  --output "${TRAINING_DATA_B}"

echo
echo "Training data A: ${TRAINING_DATA_A}"
echo "Training data B: ${TRAINING_DATA_B}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
