#!/bin/bash

set -euo pipefail

DATA_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0618-ori"
RAW_INPUT="${DATA_DIR}/msmarco-ori-80k-0618.jsonl"
FILTERED_INPUT="${DATA_DIR}/msmarco-ori-80k-0618_exclude_other_pos_save_non_pos.jsonl"
SEED="${SEED:-20260620}"

for input_file in "${RAW_INPUT}" "${FILTERED_INPUT}"; do
  if [[ ! -f "${input_file}" ]]; then
    echo "Missing input file: ${input_file}" >&2
    exit 1
  fi
done

echo "Data directory: ${DATA_DIR}"
echo "Random seed  : ${SEED}"

python - "${RAW_INPUT}" "${FILTERED_INPUT}" "${DATA_DIR}" "${SEED}" <<'PY'
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path


raw_path = Path(sys.argv[1])
filtered_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
seed = int(sys.argv[4])
sizes = (10_000, 20_000, 40_000)


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    queries: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"Expected JSON object at {path}:{line_number}")
            query = record.get("query")
            if not isinstance(query, str) or not query:
                raise SystemExit(f"Missing or invalid query at {path}:{line_number}")
            lines.append(line.rstrip("\n") + "\n")
            queries.append(query)
    return lines, queries


raw_lines, raw_queries = load_jsonl(raw_path)
filtered_lines, filtered_queries = load_jsonl(filtered_path)

if len(raw_lines) != len(filtered_lines):
    raise SystemExit(
        f"Input row counts differ: raw={len(raw_lines)}, filtered={len(filtered_lines)}"
    )

for index, (raw_query, filtered_query) in enumerate(
    zip(raw_queries, filtered_queries), start=1
):
    if raw_query != filtered_query:
        raise SystemExit(
            f"Query mismatch at logical row {index}: "
            f"raw={raw_query!r}, filtered={filtered_query!r}"
        )

if max(sizes) > len(raw_lines):
    raise SystemExit(
        f"Largest sample ({max(sizes)}) exceeds input rows ({len(raw_lines)})"
    )

indices = list(range(len(raw_lines)))
random.Random(seed).shuffle(indices)

output_dir.mkdir(parents=True, exist_ok=True)
temp_dir = Path(tempfile.mkdtemp(prefix=".nested-samples-", dir=output_dir))
outputs: list[tuple[Path, Path]] = []
try:
    for size in sizes:
        size_label = f"{size // 1000}k"
        raw_name = f"msmarco-ori-{size_label}-0618.jsonl"
        filtered_name = (
            f"msmarco-ori-{size_label}-0618_"
            "exclude_other_pos_save_non_pos.jsonl"
        )
        raw_temp = temp_dir / raw_name
        filtered_temp = temp_dir / filtered_name

        selected_indices = indices[:size]
        with (
            raw_temp.open("w", encoding="utf-8") as raw_output,
            filtered_temp.open("w", encoding="utf-8") as filtered_output,
        ):
            for index in selected_indices:
                raw_output.write(raw_lines[index])
                filtered_output.write(filtered_lines[index])

        outputs.append((raw_temp, output_dir / raw_name))
        outputs.append((filtered_temp, output_dir / filtered_name))

    # All files are complete and validated before any destination is replaced.
    for source, destination in outputs:
        os.replace(source, destination)
        print(f"Wrote {sum(1 for _ in destination.open(encoding='utf-8')):,} rows -> {destination}")
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)

print("Verified: 10k is nested in 20k, and 20k is nested in 40k.")
print("Verified: raw and filtered queries are aligned row by row.")
PY
