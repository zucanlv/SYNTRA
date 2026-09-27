#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PREPROCESS_DIR="${PROJECT_ROOT}/Main_Pipeline/corpus_preprocess"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

ANNOTATED_FILE="${PROJECT_ROOT}/Main_Pipeline/Results/cosqa_20260618_183255/Annotated_Main_20260618_191737.json"
DIVERSE_QUERY_FILE="${PROJECT_ROOT}/Main_Pipeline/Results/cosqa_20260618_183255/DiverseQuery_Main_20260618_191701.json"
OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task"

TRAINING_DATA_A="${OUTPUT_DIR}/cosqa-0618.jsonl"
TRAINING_DATA_B="${OUTPUT_DIR}/cosqa-0618_exclude_other_pos_save_non_pos.jsonl"
PROMPT="Given a natural language programming query, retrieve relevant code snippets that correctly implement, solve, or answer the query."

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
