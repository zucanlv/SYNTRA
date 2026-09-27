#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
CONFIG_ROOT="${PIPELINE_DIR}/config/RQ-2-generalization/reasoning-intensive"

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs/spartqa-winogrande}"
LOG_DIR="${LOG_ROOT}/${RUN_TS}"
Q2P_LOCK_PATH="${Q2P_LOCK_PATH:-/tmp/dsa-q2p-gpu4-5-spartqa-winogrande.lock}"
SPARTQA_FAISS_DIR="${SPARTQA_FAISS_DIR:-/data/share/project/shared_datasets/DSA/faiss/spartqa-mchoice}"
WINOGRANDE_FAISS_DIR="${WINOGRANDE_FAISS_DIR:-/data/share/project/shared_datasets/DSA/faiss/winogrande}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-30}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

# Cross-dataset order keeps runs for the same task at least two launch slots
# apart. main.py result directory names have second-level timestamp precision.
CONFIGS=(
  "${CONFIG_ROOT}/config_0731-spartqa-mchoice-gen-query.yaml"
  "${CONFIG_ROOT}/config_0731-winogrande-gen-query.yaml"
  "${CONFIG_ROOT}/config_0731-spartqa-mchoice-train-qpos.yaml"
  "${CONFIG_ROOT}/config_0731-winogrande-train-qpos.yaml"
)

NAMES=(
  "spartqa-mchoice-gen-query"
  "winogrande-gen-query"
  "spartqa-mchoice-train-qpos"
  "winogrande-train-qpos"
)

# Keep loopback traffic off any configured proxy without changing external
# OpenRouter proxy settings inherited from the caller.
export NO_PROXY="127.0.0.1,localhost,0.0.0.0${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost,0.0.0.0${no_proxy:+,${no_proxy}}"
export PYTHONUNBUFFERED=1

if [[ ! "${START_STAGGER_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "START_STAGGER_SECONDS must be a non-negative integer." >&2
  exit 2
fi

if [[ -z "${Q2P_LOCK_PATH}" ]]; then
  echo "Q2P_LOCK_PATH must not be empty." >&2
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

for faiss_dir in "${SPARTQA_FAISS_DIR}" "${WINOGRANDE_FAISS_DIR}"; do
  for artifact in index.faiss id_map.pkl doc_dict.pkl; do
    if [[ ! -f "${faiss_dir}/${artifact}" ]]; then
      echo "Missing FAISS artifact: ${faiss_dir}/${artifact}" >&2
      echo "All four runs use --skip-index, so a complete existing index is required." >&2
      exit 1
    fi
  done
done

mkdir -p "${LOG_DIR}" "$(dirname -- "${Q2P_LOCK_PATH}")"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Prepared ${#CONFIGS[@]} SpartQA/WinoGrande runs."
echo "Logs: ${LOG_DIR}"
echo "Shared Q2P lock: ${Q2P_LOCK_PATH}"
echo "Start stagger: ${START_STAGGER_SECONDS}s"

if [[ "${DRY_RUN}" == "1" ]]; then
  for i in "${!CONFIGS[@]}"; do
    command=(
      python -u main.py
      --config "${CONFIGS[$i]}"
      --skip-index
      --q2p-lock-path "${Q2P_LOCK_PATH}"
    )
    printf 'DRY_RUN command[%d]:' "$((i + 1))"
    printf ' %q' "${command[@]}"
    printf '\n'
  done
  echo "DRY_RUN complete: validated 4 commands"
  exit 0
fi

if [[ "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/ustc/public_envs/DSA
fi

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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting all four runs in parallel."

for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$i]}"
  name="${NAMES[$i]}"
  log_path="${LOG_DIR}/${name}.log"

  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name}"
    echo "Config: ${config}"
    echo "Shared Q2P lock: ${Q2P_LOCK_PATH}"
  } >"${log_path}"

  python -u main.py \
    --config "${config}" \
    --skip-index \
    --q2p-lock-path "${Q2P_LOCK_PATH}" \
    >>"${log_path}" 2>&1 &
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
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All four runs completed successfully."
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] One or more runs failed. Check ${LOG_DIR}." >&2
fi

exit "${overall_rc}"
