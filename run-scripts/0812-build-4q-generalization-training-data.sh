#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"

RESULTS_ROOT="${RESULTS_ROOT:-${PIPELINE_DIR}/Results}"
CODE_OUTPUT_DIR="${CODE_OUTPUT_DIR:-/data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/codetrans-dl}"
DOMAIN_OUTPUT_ROOT="${DOMAIN_OUTPUT_ROOT:-/data/share/project/shared_datasets/DSA/syn_data/more-domain-task}"
BUILD_DATE="${BUILD_DATE:-20260812}"
TASK_FILTER="${TASK_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

SPLIT_QUERIES="${PREPROCESS_DIR}/split_queries_by_position.py"
ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

CODETRANS_PROMPT="Given a TensorFlow code snippet, retrieve functionally equivalent PaddlePaddle code snippets that implement the same deep learning functionality."
LEGALBENCH_RAG_PROMPT="Given a question about a legal contract or privacy policy, retrieve relevant contract or policy passages that directly answer it or provide decisive evidence."
MEDICAL_QA_PROMPT="Given an English-language medical question, retrieve relevant English-language medical answer passages that directly answer the question."

TASKS=(
  "codetrans-dl"
  "legalbench-rag"
  "medical_qa"
)

RESULT_DIRS=(
  "codetrans-dl_20260811_202056"
  "legalbench-rag_20260811_202254"
  "medical_qa_20260811_202419"
)

ANNOTATED_FILENAMES=(
  "Annotated_Main_20260811_205419.json"
  "Annotated_Main_20260811_214453.json"
  "Annotated_Main_20260811_212858.json"
)

DIVERSE_QUERY_FILENAMES=(
  "DiverseQuery_Main_20260811_205055.json"
  "DiverseQuery_Main_20260811_212414.json"
  "DiverseQuery_Main_20260811_212108.json"
)

OUTPUT_DIRS=(
  "${CODE_OUTPUT_DIR}"
  "${DOMAIN_OUTPUT_ROOT}/legalbench-rag"
  "${DOMAIN_OUTPUT_ROOT}/medical_qa"
)

OUTPUT_STEMS=(
  "codetrans-dl-4q"
  "legalbench-rag-4q"
  "medical_qa-all-4q"
)

PROMPTS=(
  "${CODETRANS_PROMPT}"
  "${LEGALBENCH_RAG_PROMPT}"
  "${MEDICAL_QA_PROMPT}"
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

task_selected() {
  local task="$1"
  [[ -z "${TASK_FILTER}" || "${TASK_FILTER}" == "${task}" ]]
}

array_length="${#TASKS[@]}"
if [[ "${array_length}" -ne "${#RESULT_DIRS[@]}" ]] \
  || [[ "${array_length}" -ne "${#ANNOTATED_FILENAMES[@]}" ]] \
  || [[ "${array_length}" -ne "${#DIVERSE_QUERY_FILENAMES[@]}" ]] \
  || [[ "${array_length}" -ne "${#OUTPUT_DIRS[@]}" ]] \
  || [[ "${array_length}" -ne "${#OUTPUT_STEMS[@]}" ]] \
  || [[ "${array_length}" -ne "${#PROMPTS[@]}" ]]; then
  echo "Internal error: build arrays have different lengths." >&2
  exit 2
fi

require_file "${SPLIT_QUERIES}"
require_file "${ANNO_TO_MSMARCO}"
require_file "${EXCLUDE_OTHER_POS}"

selected_tasks=0
for i in "${!TASKS[@]}"; do
  if ! task_selected "${TASKS[$i]}"; then
    continue
  fi
  selected_tasks=$((selected_tasks + 1))
  require_file "${RESULTS_ROOT}/${RESULT_DIRS[$i]}/${ANNOTATED_FILENAMES[$i]}"
  require_file "${RESULTS_ROOT}/${RESULT_DIRS[$i]}/${DIVERSE_QUERY_FILENAMES[$i]}"
done

if [[ "${selected_tasks}" -eq 0 ]]; then
  echo "No task matched TASK_FILTER=${TASK_FILTER}" >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "1" && "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/ustc/public_envs/DSA
fi

build_count=0
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  if ! task_selected "${task}"; then
    continue
  fi

  result_dir="${RESULTS_ROOT}/${RESULT_DIRS[$i]}"
  annotated_file="${result_dir}/${ANNOTATED_FILENAMES[$i]}"
  diverse_query_file="${result_dir}/${DIVERSE_QUERY_FILENAMES[$i]}"
  output_dir="${OUTPUT_DIRS[$i]}"
  output_stem="${OUTPUT_STEMS[$i]}"
  prompt="${PROMPTS[$i]}"

  run_command mkdir -p "${output_dir}"

  for query_position in 1 2 3 4; do
    split_annotated="${annotated_file%.json}_query_position_${query_position}.json"
    split_diverse_query="${diverse_query_file%.json}_query_position_${query_position}.json"
    training_data="${output_dir}/${output_stem}-query${query_position}-${BUILD_DATE}.jsonl"
    filtered_training_data="${output_dir}/${output_stem}-query${query_position}-${BUILD_DATE}_exclude_other_pos_save_non_pos.jsonl"

    echo
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building ${task} query position ${query_position}"
    echo "  annotated    : ${split_annotated}"
    echo "  diverse query: ${split_diverse_query}"
    echo "  prompt       : ${prompt}"
    echo "  output       : ${training_data}"
    echo "  filtered     : ${filtered_training_data}"

    run_command \
      python "${SPLIT_QUERIES}" \
      --input "${annotated_file}" \
      --output "${split_annotated}" \
      --query-position "${query_position}"

    run_command \
      python "${SPLIT_QUERIES}" \
      --input "${diverse_query_file}" \
      --output "${split_diverse_query}" \
      --query-position "${query_position}"

    run_command \
      python "${ANNO_TO_MSMARCO}" \
      --input "${split_annotated}" \
      --output "${training_data}" \
      --prompt "${prompt}"

    run_command \
      python "${EXCLUDE_OTHER_POS}" \
      --input-jsonl "${training_data}" \
      --diverse-query "${split_diverse_query}" \
      --output "${filtered_training_data}"

    build_count=$((build_count + 1))
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN complete: validated ${build_count} query-position builds"
else
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ${build_count} query-position builds."
fi
