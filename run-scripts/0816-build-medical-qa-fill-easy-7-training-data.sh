#!/usr/bin/env bash

set -euo pipefail

if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/ustc/public_envs/DSA
fi

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"

ANNO_FILL_EASY="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn_fill_easy.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

RUN_1Q="${PIPELINE_DIR}/Results/medical_qa_20260806_070242"
RUN_2Q="${PIPELINE_DIR}/Results/medical_qa_20260806_070404"
RUN_4Q="${PIPELINE_DIR}/Results/medical_qa_20260811_202419"

OUTPUT_DIR="${OUTPUT_DIR:-/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medical_qa/fill-easy-7}"
PROMPT="Given an English-language medical question, retrieve relevant English-language medical answer passages that directly answer the question."
TARGET_NEGATIVES=7

ANNOTATED_FILES=(
  "${RUN_1Q}/Annotated_Main_20260806_090415.json"
  "${RUN_2Q}/Annotated_Main_20260806_090558_query_position_1.json"
  "${RUN_2Q}/Annotated_Main_20260806_090558_query_position_2.json"
  "${RUN_4Q}/Annotated_Main_20260811_212858_query_position_1.json"
  "${RUN_4Q}/Annotated_Main_20260811_212858_query_position_2.json"
  "${RUN_4Q}/Annotated_Main_20260811_212858_query_position_3.json"
  "${RUN_4Q}/Annotated_Main_20260811_212858_query_position_4.json"
)

DIVERSE_QUERY_FILES=(
  "${RUN_1Q}/DiverseQuery_Main_20260806_090338.json"
  "${RUN_2Q}/DiverseQuery_Main_20260806_090515_query_position_1.json"
  "${RUN_2Q}/DiverseQuery_Main_20260806_090515_query_position_2.json"
  "${RUN_4Q}/DiverseQuery_Main_20260811_212108_query_position_1.json"
  "${RUN_4Q}/DiverseQuery_Main_20260811_212108_query_position_2.json"
  "${RUN_4Q}/DiverseQuery_Main_20260811_212108_query_position_3.json"
  "${RUN_4Q}/DiverseQuery_Main_20260811_212108_query_position_4.json"
)

OUTPUT_NAMES=(
  "medical_qa-all-1q-20260806-fill-easy-7.jsonl"
  "medical_qa-all-2q-query1-20260811-fill-easy-7.jsonl"
  "medical_qa-all-2q-query2-20260811-fill-easy-7.jsonl"
  "medical_qa-all-4q-query1-20260812-fill-easy-7.jsonl"
  "medical_qa-all-4q-query2-20260812-fill-easy-7.jsonl"
  "medical_qa-all-4q-query3-20260812-fill-easy-7.jsonl"
  "medical_qa-all-4q-query4-20260812-fill-easy-7.jsonl"
)

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing input file: ${path}" >&2
    exit 1
  fi
}

run_cmd() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %s' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

require_file "${ANNO_FILL_EASY}"
require_file "${EXCLUDE_OTHER_POS}"
for input_file in "${ANNOTATED_FILES[@]}" "${DIVERSE_QUERY_FILES[@]}"; do
  require_file "${input_file}"
done

mkdir -p "${OUTPUT_DIR}"

for index in "${!OUTPUT_NAMES[@]}"; do
  raw_output="${OUTPUT_DIR}/${OUTPUT_NAMES[$index]}"
  filtered_output="${raw_output%.jsonl}_exclude_other_pos_save_non_pos.jsonl"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building ${OUTPUT_NAMES[$index]}"
  run_cmd python "${ANNO_FILL_EASY}" \
    --input "${ANNOTATED_FILES[$index]}" \
    --output "${raw_output}" \
    --prompt "${PROMPT}" \
    --target-negatives "${TARGET_NEGATIVES}"

  run_cmd python "${EXCLUDE_OTHER_POS}" \
    --input-jsonl "${raw_output}" \
    --diverse-query "${DIVERSE_QUERY_FILES[$index]}" \
    --output "${filtered_output}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Built 7 raw/filtered data pairs in ${OUTPUT_DIR}"

