#!/bin/bash

set -euo pipefail

DATA_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale"
PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
EXCLUDE_OTHER_POS="${PIPELINE_DIR}/corpus_preprocess/4-29-exclude-other-pos-save-non-pos.py"
SEED="${SEED:-20260511}"
SIZES=(10000 20000 40000 80000 160000 320000)

RAW_INPUTS=(
  "${DATA_DIR}/msmarco-gen-0k-40k.jsonl"
  "${DATA_DIR}/msmarco-gen-40k-60k.jsonl"
  "${DATA_DIR}/msmarco-gen-60k-120k.jsonl"
  "${DATA_DIR}/msmarco-gen-120k-150k.jsonl"
  "${DATA_DIR}/msmarco-gen-150k-180k.jsonl"
  "${DATA_DIR}/msmarco-gen-180k-210k.jsonl"
  "${DATA_DIR}/msmarco-gen-210k-240k.jsonl"
  "${DATA_DIR}/msmarco-gen-240k-270k.jsonl"
  "${DATA_DIR}/msmarco-gen-270k-300k.jsonl"
  "${DATA_DIR}/msmarco-gen-300k-330k.jsonl"
  "${DATA_DIR}/msmarco-gen-330k-360k.jsonl"
)

DIVERSE_QUERY_FILES=(
  "${PIPELINE_DIR}/Results/msmarco_20260508_075045/DiverseQuery_Main_20260508_160631.json"
  "${PIPELINE_DIR}/Results/msmarco_20260508_075025/DiverseQuery_Main_20260508_120033.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_063619/DiverseQuery_Main_20260509_173727.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_182643/DiverseQuery_Main_20260510_011420.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_182735/DiverseQuery_Main_20260510_011458.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_143942/DiverseQuery_FromFile_20260510_143944.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_182816/DiverseQuery_Main_20260510_011537.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_151052/DiverseQuery_FromFile_20260510_151054.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_151202/DiverseQuery_FromFile_20260510_151204.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_144239/DiverseQuery_FromFile_20260510_144241.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_144346/DiverseQuery_FromFile_20260510_144348.json"
)

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

  python - "$output" "$@" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
input_paths = [Path(p) for p in sys.argv[2:]]

count = 0
with output_path.open("w", encoding="utf-8") as out:
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"invalid JSONL in {input_path}:{line_no}: {exc}"
                    ) from exc
                out.write(line + "\n")
                count += 1

print(f"Merged {count} lines -> {output_path}")
PY
}

make_nested_samples() {
  local input="$1"
  local output_prefix="$2"
  shift 2

  python - "$input" "$output_prefix" "$SEED" "$@" <<'PY'
import random
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
output_prefix = Path(sys.argv[2])
seed = int(sys.argv[3])
sizes = [int(x) for x in sys.argv[4:]]

with input_path.open("r", encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]

total = len(lines)
max_size = max(sizes)
if max_size > total:
    raise SystemExit(
        f"largest sample size={max_size} is larger than total lines={total}: {input_path}"
    )

indices = list(range(total))
random.seed(seed)
random.shuffle(indices)

for size in sizes:
    output_path = output_prefix.with_name(f"{output_prefix.name}-{size // 1000}k.jsonl")
    with output_path.open("w", encoding="utf-8") as out:
        for idx in indices[:size]:
            out.write(lines[idx])
    print(f"Sampled nested {size} / {total} lines -> {output_path}")
PY
}

filter_from_raw() {
  local input="$1"
  local output="$2"

  python "${EXCLUDE_OTHER_POS}" \
    --input-jsonl "${input}" \
    --diverse-query "${DIVERSE_QUERY_FILES[@]}" \
    --output "${output}"
}

check_inputs "${RAW_INPUTS[@]}"
check_inputs "${DIVERSE_QUERY_FILES[@]}"

RAW_FULL="${DATA_DIR}/msmarco-gen-360k.jsonl"
FILTERED_FULL="${DATA_DIR}/msmarco-gen-360k_exclude_other_pos_save_non_pos.jsonl"

echo "Data directory: ${DATA_DIR}"
echo "Random seed   : ${SEED}"
echo "Sample sizes  : ${SIZES[*]}"

echo
echo "Building raw full JSONL and nested samples..."
merge_jsonl "${RAW_FULL}" "${RAW_INPUTS[@]}"
make_nested_samples "${RAW_FULL}" "${DATA_DIR}/msmarco-gen-nested" "${SIZES[@]}"

echo
echo "Building exclude_other_pos_save_non_pos files from raw JSONL files..."
filter_from_raw "${RAW_FULL}" "${FILTERED_FULL}"

for size in "${SIZES[@]}"; do
  size_k="$((size / 1000))k"
  raw_sample="${DATA_DIR}/msmarco-gen-nested-${size_k}.jsonl"
  filtered_sample="${DATA_DIR}/msmarco-gen-nested_exclude_other_pos_save_non_pos-${size_k}.jsonl"
  filter_from_raw "${raw_sample}" "${filtered_sample}"
done

echo
echo "Done."
