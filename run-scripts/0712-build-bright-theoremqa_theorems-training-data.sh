#!/usr/bin/env bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PREPROCESS_DIR="${PROJECT_ROOT}/Main_Pipeline/corpus_preprocess"
RESULT_DIR="${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-theoremqa_theorems_20260712_060447"
ANNOTATED_FILE="${RESULT_DIR}/Annotated_Main_20260712_060453.json"
OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task"
OUTPUT_STEM="bright-documents-theoremqa_theorems-20260712-reason-embed-query"
PROMPT="Given a Math problem, retrieve relevant theorems that help answer the problem."

TRAINING_DATA="${OUTPUT_DIR}/${OUTPUT_STEM}.jsonl"
FILTERED_TRAINING_DATA="${OUTPUT_DIR}/${OUTPUT_STEM}_exclude_other_pos_save_non_pos.jsonl"

for required_file in \
  "${ANNOTATED_FILE}" \
  "${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py" \
  "${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

python "${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py" \
  --input "${ANNOTATED_FILE}" \
  --output "${TRAINING_DATA}" \
  --prompt "${PROMPT}"

# Annotated_Main retains the original top-level document and query mapping, so it
# can be used directly to keep only the origin document when it is a positive.
python "${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py" \
  --input-jsonl "${TRAINING_DATA}" \
  --diverse-query "${ANNOTATED_FILE}" \
  --output "${FILTERED_TRAINING_DATA}"

echo "Training data: ${TRAINING_DATA}"
echo "Filtered training data: ${FILTERED_TRAINING_DATA}"

