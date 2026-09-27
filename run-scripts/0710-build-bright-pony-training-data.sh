#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PREPROCESS_DIR="${PROJECT_ROOT}/Main_Pipeline/corpus_preprocess"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task"
PONY_PROMPT="Given a Pony question, retrieve relevant passages that help answer the question."

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

build_pair() {
  local name="$1"
  local annotated_file="$2"
  local diverse_query_file="$3"

  local training_data_a="${OUTPUT_DIR}/${name}.jsonl"
  local training_data_b="${OUTPUT_DIR}/${name}_exclude_other_pos_save_non_pos.jsonl"

  for required_file in \
    "${ANNO_TO_MSMARCO}" \
    "${EXCLUDE_OTHER_POS}" \
    "${annotated_file}" \
    "${diverse_query_file}"; do
    require_file "${required_file}"
  done

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building ${name}"
  echo "  annotated    : ${annotated_file}"
  echo "  diverse query: ${diverse_query_file}"
  echo "  output A     : ${training_data_a}"
  echo "  output B     : ${training_data_b}"

  python "${ANNO_TO_MSMARCO}" \
    --input "${annotated_file}" \
    --output "${training_data_a}" \
    --prompt "${PONY_PROMPT}"

  python "${EXCLUDE_OTHER_POS}" \
    --input-jsonl "${training_data_a}" \
    --diverse-query "${diverse_query_file}" \
    --output "${training_data_b}"
}

mkdir -p "${OUTPUT_DIR}"

build_pair \
  "bright-documents-pony-20260710-gen-query" \
  "${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-pony_20260710_141332/Annotated_Main_20260710_141336.json" \
  "${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-pony_20260710_135948/DiverseQuery_FromFile_20260710_135948.json"

build_pair \
  "bright-documents-pony-20260710-reason-embed-query" \
  "${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-pony_20260710_052857/Annotated_Main_20260710_052859.json" \
  "${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-pony_20260707_013713/DiverseQuery_FromFile_20260707_013714.json"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
