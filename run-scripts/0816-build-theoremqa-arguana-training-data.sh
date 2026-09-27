#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"

THEOREMQA_OUTPUT_DIR="${THEOREMQA_OUTPUT_DIR:-/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task}"
ARGUANA_OUTPUT_DIR="${ARGUANA_OUTPUT_DIR:-/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

THEOREMQA_PROMPT="Given a Math problem, retrieve relevant theorems that help answer the problem."
ARGUANA_PROMPT="Given a claim, find documents that refute the claim."

NAMES=(
  "bright-documents-theoremqa_theorems-10k-20260816-wo-self-refine"
  "arguana-20260816-wo-self-refine"
  "arguana-20260816-with-self-refine"
)

ANNOTATED_FILES=(
  "${PIPELINE_DIR}/Results/bright-documents-theoremqa_theorems_20260815_182118/Annotated_Main_20260815_182128.json"
  "${PIPELINE_DIR}/Results/arguana_20260816_015519/Annotated_Main_20260816_015524.json"
  "${PIPELINE_DIR}/Results/arguana_20260816_092802/Annotated_Main_20260816_092808.json"
)

DIVERSE_QUERY_FILES=(
  "${PIPELINE_DIR}/Results/bright-documents-theoremqa_theorems_20260814_213402/DiverseQuery_Main_20260814_231554.json"
  "${PIPELINE_DIR}/Results/arguana_20260814_231602/DiverseQuery_Main_20260815_011304.json"
  "${PIPELINE_DIR}/Results/arguana_20260815_011313/DiverseQuery_Main_20260815_030955.json"
)

OUTPUT_DIRS=(
  "${THEOREMQA_OUTPUT_DIR}"
  "${ARGUANA_OUTPUT_DIR}"
  "${ARGUANA_OUTPUT_DIR}"
)

PROMPTS=(
  "${THEOREMQA_PROMPT}"
  "${ARGUANA_PROMPT}"
  "${ARGUANA_PROMPT}"
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

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  annotated_file="${ANNOTATED_FILES[$i]}"
  diverse_query_file="${DIVERSE_QUERY_FILES[$i]}"
  output_dir="${OUTPUT_DIRS[$i]}"
  prompt="${PROMPTS[$i]}"
  training_data="${output_dir}/${name}.jsonl"
  filtered_training_data="${output_dir}/${name}_exclude_other_pos_save_non_pos.jsonl"

  run_command mkdir -p "${output_dir}"

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
  echo "DRY_RUN complete: validated 3 ordinary training-data builds"
else
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 3 ordinary training-data builds are ready."
fi
