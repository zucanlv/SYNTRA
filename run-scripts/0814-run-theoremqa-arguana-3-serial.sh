#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"

CONDA_ENV="${CONDA_ENV:-/home/ustc/public_envs/DSA}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

CONFIGS=(
  "${PIPELINE_DIR}/config/RQ-2-generalization/reasoning-intensive/config_0814-bright-documents-theoremqa_theorems-10k-wo-self-refine.yaml"
  "${PIPELINE_DIR}/config/RQ-2-generalization/arguana/config_0811-arguana-wo-self-refine.yaml"
  "${PIPELINE_DIR}/config/RQ-2-generalization/arguana/config_0811-arguana-all.yaml"
)

EXPERIMENT_NAMES=(
  "theoremqa_theorems 10k without self-refine"
  "ArguAna without self-refine"
  "ArguAna all"
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

for config_path in "${CONFIGS[@]}"; do
  require_file "${config_path}"
done

if [[ "${DRY_RUN}" != "1" && "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

cd "${PIPELINE_DIR}"

failure_count=0
failed_experiments=()

for index in "${!CONFIGS[@]}"; do
  run_number=$((index + 1))
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${run_number}/3: ${EXPERIMENT_NAMES[$index]}"
  echo "  config: ${CONFIGS[$index]}"

  if run_command \
      env NO_PROXY=127.0.0.1,localhost,0.0.0.0 \
      no_proxy=127.0.0.1,localhost,0.0.0.0 \
      "${PYTHON_BIN}" -u "${PIPELINE_DIR}/main.py" \
      --config "${CONFIGS[$index]}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ${run_number}/3: ${EXPERIMENT_NAMES[$index]}"
  else
    exit_code=$?
    failure_count=$((failure_count + 1))
    failed_experiments+=("${run_number}/3 ${EXPERIMENT_NAMES[$index]} (exit ${exit_code})")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed ${run_number}/3: ${EXPERIMENT_NAMES[$index]} (exit ${exit_code}); continuing." >&2
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN complete: validated 3 serial runs"
elif (( failure_count > 0 )); then
  echo >&2
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished all 3 serial experiments: ${failure_count} failed." >&2
  printf '  - %s\n' "${failed_experiments[@]}" >&2
  exit 1
else
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 3 serial experiments completed."
fi
