#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"

ANNOTATED_PART1="${ANNOTATED_PART1:-${PIPELINE_DIR}/Results/legalbench-rag_20260812_193931/Annotated_Main_20260812_194010.json}"
ANNOTATED_PART2A="${ANNOTATED_PART2A:-${PIPELINE_DIR}/Results/legalbench-rag_20260813_020733/Annotated_Main_20260813_020804.json}"
ANNOTATED_PART2B="${ANNOTATED_PART2B:-${PIPELINE_DIR}/Results/legalbench-rag_20260813_152006/Annotated_Main_20260813_152151.json}"
DIVERSE_QUERY_FILE="${DIVERSE_QUERY_FILE:-${PIPELINE_DIR}/Results/legalbench-rag_20260811_202254/DiverseQuery_Main_20260811_212414.json}"
MERGED_ANNOTATED_FILE="${MERGED_ANNOTATED_FILE:-${PIPELINE_DIR}/Results/legalbench-rag_20260811_202254/Annotated_Main_20260813_merged_parts.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag}"
BUILD_DATE="${BUILD_DATE:-20260813}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

MERGE_PIPELINE_JSON="${PREPROCESS_DIR}/merge_pipeline_json.py"
SPLIT_QUERIES="${PREPROCESS_DIR}/split_queries_by_position.py"
ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

PROMPT="Given a question about a legal contract or privacy policy, retrieve relevant contract or policy passages that directly answer it or provide decisive evidence."

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

for path in \
  "${ANNOTATED_PART1}" \
  "${ANNOTATED_PART2A}" \
  "${ANNOTATED_PART2B}" \
  "${DIVERSE_QUERY_FILE}" \
  "${MERGE_PIPELINE_JSON}" \
  "${SPLIT_QUERIES}" \
  "${ANNO_TO_MSMARCO}" \
  "${EXCLUDE_OTHER_POS}"; do
  require_file "${path}"
done

if [[ "${DRY_RUN}" != "1" && "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/ustc/public_envs/DSA
fi

run_command mkdir -p "${OUTPUT_DIR}"

if [[ -f "${MERGED_ANNOTATED_FILE}" ]]; then
  echo "Reusing merged annotated file: ${MERGED_ANNOTATED_FILE}"
else
  echo "Merging annotated parts in source order: part1 -> part2a -> part2b"
  run_command \
    python "${MERGE_PIPELINE_JSON}" \
    --output "${MERGED_ANNOTATED_FILE}" \
    "${ANNOTATED_PART1}" \
    "${ANNOTATED_PART2A}" \
    "${ANNOTATED_PART2B}"
fi

build_count=0
for query_position in 1 2 3 4; do
  split_annotated="${MERGED_ANNOTATED_FILE%.json}_query_position_${query_position}.json"
  split_diverse_query="${DIVERSE_QUERY_FILE%.json}_query_position_${query_position}.json"
  training_data="${OUTPUT_DIR}/legalbench-rag-4q-query${query_position}-${BUILD_DATE}.jsonl"
  filtered_training_data="${OUTPUT_DIR}/legalbench-rag-4q-query${query_position}-${BUILD_DATE}_exclude_other_pos_save_non_pos.jsonl"

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building legalbench-rag query position ${query_position}"
  echo "  annotated    : ${split_annotated}"
  echo "  diverse query: ${split_diverse_query}"
  echo "  output       : ${training_data}"
  echo "  filtered     : ${filtered_training_data}"

  run_command \
    python "${SPLIT_QUERIES}" \
    --input "${MERGED_ANNOTATED_FILE}" \
    --output "${split_annotated}" \
    --query-position "${query_position}"

  run_command \
    python "${SPLIT_QUERIES}" \
    --input "${DIVERSE_QUERY_FILE}" \
    --output "${split_diverse_query}" \
    --query-position "${query_position}"

  run_command \
    python "${ANNO_TO_MSMARCO}" \
    --input "${split_annotated}" \
    --output "${training_data}" \
    --prompt "${PROMPT}"

  run_command \
    python "${EXCLUDE_OTHER_POS}" \
    --input-jsonl "${training_data}" \
    --diverse-query "${split_diverse_query}" \
    --output "${filtered_training_data}"

  build_count=$((build_count + 1))
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN complete: validated ${build_count} query-position builds"
else
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ${build_count} legalbench-rag query-position builds."
fi
