#!/bin/bash

set -euo pipefail

DATA_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0618-ori"
SEED="${SEED:-20260621}"

OLD_RAW="${DATA_DIR}/msmarco-ori-80k-0618.jsonl"
OLD_FILTERED="${DATA_DIR}/msmarco-ori-80k-0618_exclude_other_pos_save_non_pos.jsonl"
NEW_RAW="${DATA_DIR}/msmarco-ori-13804-0620.jsonl"
NEW_FILTERED="${DATA_DIR}/msmarco-ori-13804-0620_exclude_other_pos_save_non_pos.jsonl"
ARCHIVE_RAW="${DATA_DIR}/msmarco-ori-70476-0618.jsonl"
ARCHIVE_FILTERED="${DATA_DIR}/msmarco-ori-70476-0618_exclude_other_pos_save_non_pos.jsonl"

for path in "${OLD_RAW}" "${OLD_FILTERED}" "${NEW_RAW}" "${NEW_FILTERED}"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 1; }
done
for path in "${ARCHIVE_RAW}" "${ARCHIVE_FILTERED}"; do
  [[ ! -e "${path}" ]] || { echo "Archive already exists: ${path}" >&2; exit 1; }
done

python - "${OLD_RAW}" "${OLD_FILTERED}" "${NEW_RAW}" "${NEW_FILTERED}" \
  "${ARCHIVE_RAW}" "${ARCHIVE_FILTERED}" "${SEED}" <<'PY'
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

old_raw_path, old_filtered_path, new_raw_path, new_filtered_path, archive_raw, archive_filtered = map(Path, sys.argv[1:7])
seed = int(sys.argv[7])
sample_size = 9_524

def load(path):
    lines, records = [], []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("query"), str):
                raise SystemExit(f"Invalid record at {path}:{line_no}")
            lines.append(line.rstrip("\n") + "\n")
            records.append(record)
    return lines, records

def validate_pair(label, raw, filtered):
    if len(raw) != len(filtered):
        raise SystemExit(f"{label} counts differ: {len(raw)} != {len(filtered)}")
    for line_no, (left, right) in enumerate(zip(raw, filtered), 1):
        if left["query"] != right["query"]:
            raise SystemExit(f"{label} query mismatch at row {line_no}")
        left_other = {k: v for k, v in left.items() if k != "pos"}
        right_other = {k: v for k, v in right.items() if k != "pos"}
        if left_other != right_other:
            raise SystemExit(f"{label} non-pos fields differ at row {line_no}")

old_raw_lines, old_raw = load(old_raw_path)
old_filtered_lines, old_filtered = load(old_filtered_path)
new_raw_lines, new_raw = load(new_raw_path)
new_filtered_lines, new_filtered = load(new_filtered_path)
validate_pair("old", old_raw, old_filtered)
validate_pair("new", new_raw, new_filtered)

if len(old_raw) != 70_476:
    raise SystemExit(f"Expected 70,476 old rows, found {len(old_raw)}")
if len(new_raw) != 12_311:
    raise SystemExit(f"Expected 12,311 new rows, found {len(new_raw)}")

indices = random.Random(seed).sample(range(len(new_raw)), sample_size)
combined_raw = old_raw_lines + [new_raw_lines[i] for i in indices]
combined_filtered = old_filtered_lines + [new_filtered_lines[i] for i in indices]
if len(combined_raw) != 80_000 or len(combined_filtered) != 80_000:
    raise SystemExit("Combined output is not exactly 80,000 rows")

temp_dir = Path(tempfile.mkdtemp(prefix=".true-80k-", dir=old_raw_path.parent))
temp_raw = temp_dir / old_raw_path.name
temp_filtered = temp_dir / old_filtered_path.name
try:
    temp_raw.write_text("".join(combined_raw), encoding="utf-8")
    temp_filtered.write_text("".join(combined_filtered), encoding="utf-8")
    os.replace(old_raw_path, archive_raw)
    os.replace(old_filtered_path, archive_filtered)
    os.replace(temp_raw, old_raw_path)
    os.replace(temp_filtered, old_filtered_path)
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)

print(f"Archived original raw      : {archive_raw}")
print(f"Archived original filtered : {archive_filtered}")
print(f"Sampled new rows           : {sample_size:,} / {len(new_raw):,}")
print(f"True 80k raw               : {old_raw_path}")
print(f"True 80k filtered          : {old_filtered_path}")
PY
