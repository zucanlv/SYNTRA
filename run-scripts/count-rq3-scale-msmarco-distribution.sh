#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
DATA_ROOT="/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale"
OUT_ROOT="${REPO_ROOT}/Main_Pipeline/corpus_preprocess/rq3-scale-msmarco-distribution"
TEXT_DIR="${OUT_ROOT}/text"
PNG_DIR="${OUT_ROOT}/png"

POS_NEG_SCRIPT="${REPO_ROOT}/Main_Pipeline/corpus_preprocess/4-21-count-training-data-pos-neg.py"
QUERY_LEN_SCRIPT="${REPO_ROOT}/Main_Pipeline/corpus_preprocess/4-27-compile-training-set-query-length.py"

mkdir -p "${TEXT_DIR}" "${PNG_DIR}"

DATASETS=(
  "${DATA_ROOT}/msmarco-gen-nested_exclude_other_pos_save_non_pos-10k.jsonl"
  "${DATA_ROOT}/msmarco-gen-nested_exclude_other_pos_save_non_pos-20k.jsonl"
  "${DATA_ROOT}/msmarco-gen-nested_exclude_other_pos_save_non_pos-40k.jsonl"
  "${DATA_ROOT}/msmarco-gen-nested_exclude_other_pos_save_non_pos-80k.jsonl"
  "${DATA_ROOT}/msmarco-gen-nested_exclude_other_pos_save_non_pos-160k.jsonl"
  "${DATA_ROOT}/msmarco-gen-nested_exclude_other_pos_save_non_pos-320k.jsonl"
  "${DATA_ROOT}/msmarco-gen-nested-10k.jsonl"
  "${DATA_ROOT}/msmarco-gen-nested-20k.jsonl"
  "${DATA_ROOT}/msmarco-gen-nested-40k.jsonl"
  "${DATA_ROOT}/msmarco-gen-nested-80k.jsonl"
  "${DATA_ROOT}/msmarco-gen-nested-160k.jsonl"
  "${DATA_ROOT}/msmarco-gen-nested-320k.jsonl"
)

for data_path in "${DATASETS[@]}"; do
  name="$(basename "${data_path}" .jsonl)"
  echo "===== ${name} ====="

  python "${POS_NEG_SCRIPT}" "${data_path}" \
    --save-png "${PNG_DIR}" \
    > "${TEXT_DIR}/${name}.pos_neg.txt"

  python "${QUERY_LEN_SCRIPT}" "${data_path}" \
    --output-dir "${PNG_DIR}" \
    > "${TEXT_DIR}/${name}.query_length.txt"
done

echo
echo "Text reports: ${TEXT_DIR}"
echo "PNG reports : ${PNG_DIR}"
