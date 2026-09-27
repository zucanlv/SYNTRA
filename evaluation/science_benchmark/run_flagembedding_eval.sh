#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'USAGE'
Usage:
  bash run_flagembedding_eval.sh <beir_task_dir> <model_name_or_path> <output_dir> [extra args...]

Example:
  bash run_flagembedding_eval.sh data/converted/scifact BAAI/bge-base-en-v1.5 outputs/scifact

The task directory must contain:
  corpus.jsonl
  queries.jsonl
  qrels/test.tsv
USAGE
  exit 2
fi

TASK_DIR="$1"
MODEL_NAME_OR_PATH="$2"
OUTPUT_DIR="$3"
shift 3

PYTHON_BIN="${PYTHON:-/data/share/project/public_envs/embedder_train_eval/bin/python}"

"$PYTHON_BIN" -m flagembedding_eval.evaluate_beir \
  --data-dir "$TASK_DIR" \
  --model-name-or-path "$MODEL_NAME_OR_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
