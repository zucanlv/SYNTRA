#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
BUILD_QPOS="${PROJECT_ROOT}/Main_Pipeline/corpus_preprocess/build_qpos_from_diverse_texts.py"

DIVERSE_TEXTS="/data/share/project/shared_datasets/DSA/others/diverse_texts/diverse_texts_cos062_training_corpus_499996_93804_unsampled13804.jsonl"
TRAINING_SET="/data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/MSMARCO/msmarco_hn_train.jsonl"
OUTPUT="/data/share/project/shared_datasets/DSA/others/qpos/diverse_texts_cos062_training_corpus_499996_93804_unsampled13804_qpos_new.json"
MISSING_OUTPUT="/data/share/project/shared_datasets/DSA/others/qpos/diverse_texts_cos062_training_corpus_499996_93804_unsampled13804_qpos_missing.jsonl"

for required_file in "${BUILD_QPOS}" "${DIVERSE_TEXTS}" "${TRAINING_SET}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building qpos for all 13,804 unsampled docs"
python "${BUILD_QPOS}" \
  --diverse-texts "${DIVERSE_TEXTS}" \
  --training-set "${TRAINING_SET}" \
  --output "${OUTPUT}" \
  --max-queries-per-doc 1 \
  --missing-output "${MISSING_OUTPUT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] qpos output: ${OUTPUT}"
if [[ -f "${MISSING_OUTPUT}" ]]; then
  echo "Unmatched docs: ${MISSING_OUTPUT}"
fi
