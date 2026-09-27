#!/bin/bash

set -u
set -o pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

# Keep local OpenAI-compatible Qwen server traffic off the proxy while allowing
# external OpenRouter stages to use the current proxy environment.
export NO_PROXY=127.0.0.1,localhost,0.0.0.0
export no_proxy=127.0.0.1,localhost,0.0.0.0

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
LOG_DIR="${PROJECT_ROOT}/run-scripts/logs"
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
START_STAGGER_SECONDS=120

mkdir -p "${LOG_DIR}"
cd "${PIPELINE_DIR}" || exit 1

CONFIGS=(
  "config/RQ-1-gen-ori/config_0509-msmarco-ori-0k-24k.yaml"
  "config/RQ-1-gen-ori/config_0509-msmarco-ori-24k-48k.yaml"
  "config/RQ-1-gen-ori/config_0509-msmarco-ori-48k-74k.yaml"
)

NAMES=(
  "msmarco-ori-0k-24k"
  "msmarco-ori-24k-48k"
  "msmarco-ori-48k-74k"
)

pids=()
statuses=()

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${#CONFIGS[@]} RQ-1 gen-ori runs in parallel."

for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$i]}"
  name="${NAMES[$i]}"
  log_path="${LOG_DIR}/${RUN_TS}_${name}.log"

  (
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name}"
    echo "Config: ${config}"
    python main.py --config "${config}"
    rc=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${name} exit_code=${rc}"
    exit "${rc}"
  ) >"${log_path}" 2>&1 &

  pid=$!
  pids+=("${pid}")
  statuses+=("pending")
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched ${name} pid=${pid} log=${log_path}"

  if [[ "${i}" -lt "$((${#CONFIGS[@]} - 1))" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting ${START_STAGGER_SECONDS}s before launching next run."
    sleep "${START_STAGGER_SECONDS}"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name} is still running; launching next run."
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: ${name} exited before next launch. Check ${log_path}."
    fi
  fi
done

overall_rc=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${NAMES[$i]}"
  if wait "${pid}"; then
    statuses[$i]="ok"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: ${name}"
  else
    rc=$?
    statuses[$i]="failed:${rc}"
    overall_rc=1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed: ${name} exit_code=${rc}"
  fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Summary:"
for i in "${!NAMES[@]}"; do
  echo "  ${NAMES[$i]} -> ${statuses[$i]}"
done

if [[ "${overall_rc}" -eq 0 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done."
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Some runs failed. Check logs in ${LOG_DIR}."
fi

exit "${overall_rc}"
