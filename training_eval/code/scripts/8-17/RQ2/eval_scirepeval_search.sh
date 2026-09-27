#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'USAGE'
Usage:
  bash eval_scirepeval_search.sh <model_path> <output_dir> <raw_evaluation_parquet_or_dir> [extra FlagEmbedding args...]

The input must be the SciRepEval Search evaluation parquet, not the full SciRepEval
dataset directory. The converted custom dataset is written to <output_dir>/dataset_custom.
USAGE
  exit 2
fi

MODEL_PATH="$1"
OUTPUT_DIR="$2"
RAW_EVALUATION_PATH="$3"
shift 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)"
PYTHON_BIN="${PYTHON:-/data/share/project/public_envs/embedder_train_eval/bin/python}"
CONVERTER="$SCRIPT_DIR/convert_scirepeval_search_to_custom.py"
CUSTOM_RUNNER="$PROJECT_ROOT/science_benchmark/run_flagembedding_custom_eval.sh"
DATASET_DIR="$OUTPUT_DIR/dataset_custom"

[[ -x "$PYTHON_BIN" ]] || { echo "Python interpreter not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONVERTER" ]] || { echo "Converter not found: $CONVERTER" >&2; exit 2; }
[[ -f "$CUSTOM_RUNNER" ]] || { echo "Custom runner not found: $CUSTOM_RUNNER" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" "$CONVERTER" \
  --input "$RAW_EVALUATION_PATH" \
  --output "$DATASET_DIR" \
  --overwrite

bash "$CUSTOM_RUNNER" \
  scirepeval_search_tasks \
  "$DATASET_DIR" \
  "$MODEL_PATH" \
  "$OUTPUT_DIR" \
  "$@"
