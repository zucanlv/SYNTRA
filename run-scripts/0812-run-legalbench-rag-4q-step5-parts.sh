#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"

CONFIG_PATH="${CONFIG_PATH:-${PIPELINE_DIR}/config/RQ-2-generalization/law/legalbench-rag/config_0805-legalbench-rag-4q.yaml}"
LEGALBENCH_RESULTS_DIR="${LEGALBENCH_RESULTS_DIR:-${PIPELINE_DIR}/Results/legalbench-rag_20260811_202254}"
CONDA_ENV="${CONDA_ENV:-/home/ustc/public_envs/DSA}"
PYTHON_BIN="${PYTHON_BIN:-python}"
START_PART="${START_PART:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

MAX_STEP5_INPUT_BYTES=$((5 * 1024 * 1024 * 1024))

PART_NAMES=(
  "Query2Passage_Main_20260811_212858_part1.json"
  "Query2Passage_Main_20260811_212858_part2a.json"
  "Query2Passage_Main_20260811_212858_part2b.json"
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

if [[ ! "${START_PART}" =~ ^[1-3]$ ]]; then
  echo "START_PART must be 1, 2, or 3; got: ${START_PART}" >&2
  exit 2
fi

require_file "${CONFIG_PATH}"
for part_name in "${PART_NAMES[@]}"; do
  part_path="${LEGALBENCH_RESULTS_DIR}/${part_name}"
  require_file "${part_path}"
  part_size="$(stat -c '%s' "${part_path}")"
  if (( part_size >= MAX_STEP5_INPUT_BYTES )); then
    echo "Step 5 input is at least 5 GiB and must be split further: ${part_path}" >&2
    exit 1
  fi
done

if [[ "${DRY_RUN}" != "1" && "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

run_count=0
for index in "${!PART_NAMES[@]}"; do
  part_number=$((index + 1))
  if (( part_number < START_PART )); then
    continue
  fi

  part_path="${LEGALBENCH_RESULTS_DIR}/${PART_NAMES[$index]}"
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running legalbench-rag Step 5 part ${part_number}/3"
  echo "  input : ${part_path}"
  echo "  config: ${CONFIG_PATH}"

  run_command \
    "${PYTHON_BIN}" "${PIPELINE_DIR}/main.py" \
    --config "${CONFIG_PATH}" \
    --step5-query2passage-input "${part_path}"

  run_count=$((run_count + 1))
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN complete: validated ${run_count} Step 5 runs"
else
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ${run_count} legalbench-rag Step 5 runs."
fi
