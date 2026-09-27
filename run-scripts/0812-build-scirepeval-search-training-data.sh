#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"
OUTPUT_DIR="${OUTPUT_DIR:-/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/scirepeval-search}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

ANNOTATED_FILE="${PIPELINE_DIR}/Results/scirepeval-search_20260812_044416/Annotated_Main_20260812_044430.json"
DIVERSE_QUERY_FILE="${PIPELINE_DIR}/Results/scirepeval-search_20260812_023338/DiverseQuery_Main_20260812_041428.json"

TRAINING_DATA="${OUTPUT_DIR}/scirepeval-search-10k-20260812.jsonl"
FILTERED_TRAINING_DATA="${OUTPUT_DIR}/scirepeval-search-10k-20260812_exclude_other_pos_save_non_pos.jsonl"

PROMPT="Given a scientific literature search query, retrieve relevant scientific papers that directly address the specified research topic or satisfy the requested constraints."

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

run_command() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

for input_file in \
  "${ANNO_TO_MSMARCO}" \
  "${EXCLUDE_OTHER_POS}" \
  "${ANNOTATED_FILE}" \
  "${DIVERSE_QUERY_FILE}"; do
  require_file "${input_file}"
done

if [[ "${DRY_RUN}" != "1" && "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/ustc/public_envs/DSA
fi

run_command mkdir -p "${OUTPUT_DIR}"

echo "Prompt: ${PROMPT}"
echo "Output: ${TRAINING_DATA}"
echo "Filtered output: ${FILTERED_TRAINING_DATA}"

run_command \
  python "${ANNO_TO_MSMARCO}" \
  --input "${ANNOTATED_FILE}" \
  --output "${TRAINING_DATA}" \
  --prompt "${PROMPT}"

run_command \
  python "${EXCLUDE_OTHER_POS}" \
  --input-jsonl "${TRAINING_DATA}" \
  --diverse-query "${DIVERSE_QUERY_FILE}" \
  --output "${FILTERED_TRAINING_DATA}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN complete: validated ordinary SciRepEval Search training-data build"
else
  printf '%s\n' "${TRAINING_DATA}" "${FILTERED_TRAINING_DATA}"
fi
