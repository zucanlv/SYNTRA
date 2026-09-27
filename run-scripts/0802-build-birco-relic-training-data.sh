#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"
OUTPUT_DIR="${OUTPUT_DIR:-/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/birco-relic}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

ANNOTATED_FILE="${PIPELINE_DIR}/Results/birco-relic_20260801_233300/Annotated_Main_20260801_233302.json"
DIVERSE_QUERY_FILE="${PIPELINE_DIR}/Results/birco-relic_20260801_195801/DiverseQuery_Main_20260801_205707.json"

TRAINING_DATA="${OUTPUT_DIR}/birco-relic-20260801-gen-query.jsonl"
FILTERED_TRAINING_DATA="${OUTPUT_DIR}/birco-relic-20260801-gen-query_exclude_other_pos_save_non_pos.jsonl"

PROMPT="Given a literary analysis excerpt with a masked quotation, retrieve passages from the analyzed work that fit the gap and support the surrounding interpretation."

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

mkdir -p "${OUTPUT_DIR}"

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
  echo "DRY_RUN complete: validated ordinary BIRCO-RELIC training-data build"
else
  printf '%s\n' "${TRAINING_DATA}" "${FILTERED_TRAINING_DATA}"
fi
