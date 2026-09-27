#!/usr/bin/env bash

set -uo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
CONFIG_ROOT="${PIPELINE_DIR}/config/RQ-3-scale/scale-size-of-backend-llm"
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${SCRIPT_DIR}/logs/scale-size-of-backend-llm/${RUN_TS}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-0}"

# Keep loopback traffic off any configured proxy. External DeepSeek requests
# continue to use the caller's proxy environment when one is configured.
export NO_PROXY="127.0.0.1,localhost,0.0.0.0${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost,0.0.0.0${no_proxy:+,${no_proxy}}"
export PYTHONUNBUFFERED=1

CONFIGS=(
  "${CONFIG_ROOT}/msmarco/config-0714-msmarco-10k-deepseek-v4-flash.yaml"
  "${CONFIG_ROOT}/msmarco/config-0714-msmarco-10k-deepseek-v4-pro.yaml"
  "${CONFIG_ROOT}/synthetic-text2sql/config-0714-synthetic-text2sql-10k-deepseek-v4-flash.yaml"
  "${CONFIG_ROOT}/synthetic-text2sql/config-0714-synthetic-text2sql-10k-deepseek-v4-pro.yaml"
  "${CONFIG_ROOT}/theoremqa-theorem/config-0714-theoremqa-theorem-10k-deepseek-v4-flash.yaml"
  "${CONFIG_ROOT}/theoremqa-theorem/config-0714-theoremqa-theorem-10k-deepseek-v4-pro.yaml"
)

NAMES=(
  "msmarco-deepseek-v4-flash"
  "msmarco-deepseek-v4-pro"
  "synthetic-text2sql-deepseek-v4-flash"
  "synthetic-text2sql-deepseek-v4-pro"
  "theoremqa-theorem-deepseek-v4-flash"
  "theoremqa-theorem-deepseek-v4-pro"
)

if [[ ! "${START_STAGGER_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "START_STAGGER_SECONDS must be a non-negative integer." >&2
  exit 2
fi

if [[ "${#CONFIGS[@]}" -ne "${#NAMES[@]}" ]]; then
  echo "Internal error: CONFIGS and NAMES have different lengths." >&2
  exit 2
fi

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Missing config: ${config}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}"
cd "${PIPELINE_DIR}" || exit 1

pids=()
statuses=()

shutdown() {
  local exit_code="$1"
  trap - INT TERM
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stopping active runs..."
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  exit "${exit_code}"
}

trap 'shutdown 130' INT
trap 'shutdown 143' TERM

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${#CONFIGS[@]} backend-LLM scale runs in parallel."
echo "Logs: ${LOG_DIR}"

for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$i]}"
  name="${NAMES[$i]}"
  log_path="${LOG_DIR}/${name}.log"

  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name}"
    echo "Config: ${config}"
  } >"${log_path}"

  python -u main.py --config "${config}" >>"${log_path}" 2>&1 &
  pid=$!
  pids+=("${pid}")
  statuses+=("running")
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched ${name} pid=${pid} log=${log_path}"

  if (( START_STAGGER_SECONDS > 0 && i < ${#CONFIGS[@]} - 1 )); then
    sleep "${START_STAGGER_SECONDS}"
  fi
done

overall_rc=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${NAMES[$i]}"
  log_path="${LOG_DIR}/${name}.log"

  if wait "${pid}"; then
    rc=0
    statuses[$i]="ok"
  else
    rc=$?
    statuses[$i]="failed:${rc}"
    overall_rc=1
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${name} exit_code=${rc}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${name} exit_code=${rc}" >>"${log_path}"
done

trap - INT TERM

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Summary:"
for i in "${!NAMES[@]}"; do
  echo "  ${NAMES[$i]} -> ${statuses[$i]}"
done

if [[ "${overall_rc}" -eq 0 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All six runs completed successfully."
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] One or more runs failed. Check ${LOG_DIR}." >&2
fi

exit "${overall_rc}"
