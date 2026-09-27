#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PREPROCESS_DIR="${PROJECT_ROOT}/Main_Pipeline/corpus_preprocess"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task"

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
  local prompt="$4"

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
    --prompt "${prompt}"

  python "${EXCLUDE_OTHER_POS}" \
    --input-jsonl "${training_data_a}" \
    --diverse-query "${diverse_query_file}" \
    --output "${training_data_b}"
}

mkdir -p "${OUTPUT_DIR}"

PONY_PROMPT="Given a Pony question, retrieve relevant passages that help answer the question."
THEOREMQA_QUESTIONS_PROMPT="Given a Math problem, retrieve relevant examples that help answer the problem."

build_pair \
  "bright-documents-pony-20260707-gen-query" \
  "${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-pony_20260707_014052/Annotated_Main_20260707_014056.json" \
  "${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-pony_20260707_013531/DiverseQuery_FromFile_20260707_013532.json" \
  "${PONY_PROMPT}"

build_pair \
  "bright-documents-pony-20260707-reason-embed-query" \
  "${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-pony_20260707_063725/Annotated_Main_20260707_063726.json" \
  "${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-pony_20260707_013713/DiverseQuery_FromFile_20260707_013714.json" \
  "${PONY_PROMPT}"

build_pair \
  "bright-documents-theoremqa_questions-20260708-gen-query" \
  "${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-theoremqa_questions_20260708_041522/Annotated_Main_20260708_041527.json" \
  "${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-theoremqa_questions_20260708_040932/DiverseQuery_FromFile_20260708_040933.json" \
  "${THEOREMQA_QUESTIONS_PROMPT}"

build_pair \
  "bright-documents-theoremqa_questions-20260708-reason-embed-query" \
  "${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-theoremqa_questions_20260708_111526/Annotated_Main_20260708_111530.json" \
  "${PROJECT_ROOT}/Main_Pipeline/Results/bright-documents-theoremqa_questions_20260708_041158/DiverseQuery_FromFile_20260708_041158.json" \
  "${THEOREMQA_QUESTIONS_PROMPT}"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
