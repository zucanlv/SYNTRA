#!/bin/bash

set -euo pipefail

DATA_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori"
SEED="${SEED:-20260511}"

RAW_INPUTS=(
  "${DATA_DIR}/msmarco-gen-0k-24k.jsonl"
  "${DATA_DIR}/msmarco-gen-24k-48k.jsonl"
  "${DATA_DIR}/msmarco-gen-48k-74k.jsonl"
)

FILTERED_INPUTS=(
  "${DATA_DIR}/msmarco-gen-0k-24k_exclude_other_pos_save_non_pos.jsonl"
  "${DATA_DIR}/msmarco-gen-24k-48k_exclude_other_pos_save_non_pos.jsonl"
  "${DATA_DIR}/msmarco-gen-48k-74k_exclude_other_pos_save_non_pos.jsonl"
)

RAW_FULL="${DATA_DIR}/msmarco-gen-74k.jsonl"
RAW_HALF="${DATA_DIR}/msmarco-gen-74k_half.jsonl"
FILTERED_FULL="${DATA_DIR}/msmarco-gen-74k_exclude_other_pos_save_non_pos.jsonl"
FILTERED_HALF="${DATA_DIR}/msmarco-gen-74k_exclude_other_pos_save_non_pos_half.jsonl"

check_inputs() {
  local path
  for path in "$@"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing input file: ${path}" >&2
      exit 1
    fi
  done
}

merge_jsonl() {
  local output="$1"
  shift

  : >"${output}"
  local input
  for input in "$@"; do
    cat "${input}" >>"${output}"
  done

  local n
  n=$(wc -l <"${output}")
  echo "Merged ${n} lines -> ${output}"
}

sample_half_jsonl() {
  local input="$1"
  local output="$2"

  python - "$input" "$output" "$SEED" <<'PY'
import random
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
seed = int(sys.argv[3])

with input_path.open("r", encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]

total = len(lines)
sample_size = total // 2

random.seed(seed)
indices = sorted(random.sample(range(total), sample_size))

with output_path.open("w", encoding="utf-8") as f:
    for idx in indices:
        f.write(lines[idx])

print(f"Sampled {sample_size} / {total} lines -> {output_path}")
PY
}

check_inputs "${RAW_INPUTS[@]}"
check_inputs "${FILTERED_INPUTS[@]}"

echo "Data directory: ${DATA_DIR}"
echo "Random seed   : ${SEED}"

echo
echo "Building raw full JSONL..."
merge_jsonl "${RAW_FULL}" "${RAW_INPUTS[@]}"
sample_half_jsonl "${RAW_FULL}" "${RAW_HALF}"

echo
echo "Building exclude_other_pos_save_non_pos full JSONL..."
merge_jsonl "${FILTERED_FULL}" "${FILTERED_INPUTS[@]}"
sample_half_jsonl "${FILTERED_FULL}" "${FILTERED_HALF}"

echo
echo "Done."
