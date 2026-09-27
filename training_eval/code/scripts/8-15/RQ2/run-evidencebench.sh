#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train-qwen3_0.6b.sh"
EVAL_SCRIPT="/data/share/project/tr/science_benchmark/run_flagembedding_eval.sh"

FILTERED_DATA="/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/evidencebench/evidencebench-10k-1q-20260806_exclude_other_pos_save_non_pos.jsonl"
RAW_DATA="/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/evidencebench/evidencebench-10k-1q-20260806.jsonl"
TRAIN_DATA="${FILTERED_DATA} ${RAW_DATA}"

MODEL_OUTPUT="/data/share/project/tr/data_synthesis_agent/model_pth/8-11/RQ2/evidencebench/train-qwen3_0.6b-evidencebench"
TRAIN_LOG="/data/share/project/tr/data_synthesis_agent/logs/8-11/RQ2/evidencebench/train-qwen3_0.6b-evidencebench.log"
EVAL_DATA="/data/share/project/tr/science_benchmark/converted_full/evidencebench"
EVAL_OUTPUT="/data/share/project/tr/data_synthesis_agent/eval_output/8-11-evidencebench/RQ2/evidencebench/train-qwen3_0.6b-evidencebench"

for required_path in \
  "${TRAIN_SCRIPT}" \
  "${EVAL_SCRIPT}" \
  "${FILTERED_DATA}" \
  "${RAW_DATA}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required file: ${required_path}" >&2
    exit 1
  fi
done

if [[ ! -d "${EVAL_DATA}" ]]; then
  echo "Missing EvidenceBench evaluation directory: ${EVAL_DATA}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${TRAIN_LOG}")" "${EVAL_OUTPUT}"

echo "Training EvidenceBench model with:"
printf '  %s\n' "${FILTERED_DATA}" "${RAW_DATA}"

bash "${TRAIN_SCRIPT}" \
  "${MODEL_OUTPUT}" \
  "${TRAIN_DATA}" 2>&1 | tee "${TRAIN_LOG}"

bash "${EVAL_SCRIPT}" \
  "${EVAL_DATA}" \
  "${MODEL_OUTPUT}/merged_model" \
  "${EVAL_OUTPUT}" \
  --model-class decoder-only-base \
  --query-instruction-format $'Instruct: {}\nQuery: {}'
