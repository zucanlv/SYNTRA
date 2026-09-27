#!/usr/bin/env bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

ANNOTATED_FILE="${PIPELINE_DIR}/Results/medicalretrieval_20260729_202905/Annotated_Main_20260729_202908.json"
DIVERSE_QUERY_FILE="${PIPELINE_DIR}/Results/medicalretrieval_20260729_144402/DiverseQuery_Main_20260729_154219.json"

OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medicalretrieval"
TRAINING_DATA="${OUTPUT_DIR}/medicalretrieval-10k-0730.jsonl"
FILTERED_TRAINING_DATA="${OUTPUT_DIR}/medicalretrieval-10k-0730_exclude_other_pos_save_non_pos.jsonl"

PROMPT="Given a Chinese-language medical or health question, retrieve relevant Chinese-language medical answer passages that directly answer or help resolve the health concern."

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing input file: ${path}" >&2
    exit 1
  fi
}

for input_file in \
  "${ANNO_TO_MSMARCO}" \
  "${EXCLUDE_OTHER_POS}" \
  "${ANNOTATED_FILE}" \
  "${DIVERSE_QUERY_FILE}"; do
  require_file "${input_file}"
done

mkdir -p "${OUTPUT_DIR}"

python "${ANNO_TO_MSMARCO}" \
  --input "${ANNOTATED_FILE}" \
  --output "${TRAINING_DATA}" \
  --prompt "${PROMPT}"

python "${EXCLUDE_OTHER_POS}" \
  --input-jsonl "${TRAINING_DATA}" \
  --diverse-query "${DIVERSE_QUERY_FILE}" \
  --output "${FILTERED_TRAINING_DATA}"

printf '%s\n' "${TRAINING_DATA}" "${FILTERED_TRAINING_DATA}"
