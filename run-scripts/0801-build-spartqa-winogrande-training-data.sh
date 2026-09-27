#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"
OUTPUT_DIR="${OUTPUT_DIR:-/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

SPARTQA_PROMPT="Given a scene description and a multiple-choice spatial question, retrieve the candidate answer option supported by the stated relations and applicable spatial inference rules, including container-to-content relation inheritance."
WINOGRANDE_PROMPT="Given a sentence with a blank marked by an underscore, retrieve the answer phrase that yields the intended commonsense-plausible completion when inserted into the blank."

NAMES=(
  "spartqa-mchoice-20260801-gen-query"
  "spartqa-mchoice-20260801-ori-query"
  "winogrande-20260801-gen-query"
  "winogrande-20260801-ori-query"
)

ANNOTATED_FILES=(
  "${PIPELINE_DIR}/Results/spartqa-mchoice_20260801_050243/Annotated_Main_20260801_050243.json"
  "${PIPELINE_DIR}/Results/spartqa-mchoice_20260731_210619/Annotated_Main_20260731_210630.json"
  "${PIPELINE_DIR}/Results/winogrande_20260801_050246/Annotated_Main_20260801_050247.json"
  "${PIPELINE_DIR}/Results/winogrande_20260801_050257/Annotated_Main_20260801_050258.json"
)

DIVERSE_QUERY_FILES=(
  "${PIPELINE_DIR}/Results/spartqa-mchoice_20260731_233531/DiverseQuery_Main_20260731_235706.json"
  "${PIPELINE_DIR}/Results/spartqa-mchoice_20260731_210619/DiverseQuery_FromFile_20260731_210619.json"
  "${PIPELINE_DIR}/Results/winogrande_20260731_233631/DiverseQuery_Main_20260801_000232.json"
  "${PIPELINE_DIR}/Results/winogrande_20260801_045658/DiverseQuery_FromFile_20260801_045658.json"
)

PROMPTS=(
  "${SPARTQA_PROMPT}"
  "${SPARTQA_PROMPT}"
  "${WINOGRANDE_PROMPT}"
  "${WINOGRANDE_PROMPT}"
)

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

if [[ "${#NAMES[@]}" -ne "${#ANNOTATED_FILES[@]}" ]] \
  || [[ "${#NAMES[@]}" -ne "${#DIVERSE_QUERY_FILES[@]}" ]] \
  || [[ "${#NAMES[@]}" -ne "${#PROMPTS[@]}" ]]; then
  echo "Internal error: build arrays have different lengths." >&2
  exit 2
fi

require_file "${ANNO_TO_MSMARCO}"
require_file "${EXCLUDE_OTHER_POS}"
for i in "${!NAMES[@]}"; do
  require_file "${ANNOTATED_FILES[$i]}"
  require_file "${DIVERSE_QUERY_FILES[$i]}"
done

if [[ "${DRY_RUN}" != "1" && "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/ustc/public_envs/DSA
fi

mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  annotated_file="${ANNOTATED_FILES[$i]}"
  diverse_query_file="${DIVERSE_QUERY_FILES[$i]}"
  prompt="${PROMPTS[$i]}"
  training_data="${OUTPUT_DIR}/${name}.jsonl"
  filtered_training_data="${OUTPUT_DIR}/${name}_exclude_other_pos_save_non_pos.jsonl"

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building ${name}"
  echo "  annotated    : ${annotated_file}"
  echo "  diverse query: ${diverse_query_file}"
  echo "  prompt       : ${prompt}"
  echo "  output       : ${training_data}"
  echo "  filtered     : ${filtered_training_data}"

  run_command \
    python "${ANNO_TO_MSMARCO}" \
    --input "${annotated_file}" \
    --output "${training_data}" \
    --prompt "${prompt}"

  run_command \
    python "${EXCLUDE_OTHER_POS}" \
    --input-jsonl "${training_data}" \
    --diverse-query "${diverse_query_file}" \
    --output "${filtered_training_data}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN complete: validated 4 ordinary training-data builds"
else
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All four ordinary training-data builds are ready."
fi
