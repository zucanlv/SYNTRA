#!/usr/bin/env bash
# Evaluate BGE-M3 with the original allenai/scirepeval runner.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'USAGE'
Usage:
  SCIREPEVAL_REPO=/path/to/scirepeval \
  HF_DATASETS_CACHE=/path/to/huggingface/datasets/cache \
  bash eval_scirepeval_official_bge_m3.sh <bge_m3_model_path_or_hf_id> <output_dir> [task name ...]

Examples:
  # Run only the official Search task.
  ... eval_scirepeval_official_bge_m3.sh /models/bge-m3 outputs/bge-m3 Search

  # Run every task in scirepeval_tasks.jsonl.
  ... eval_scirepeval_official_bge_m3.sh BAAI/bge-m3 outputs/bge-m3
USAGE
  exit 2
fi

MODEL_INPUT="$1"
OUTPUT_DIR="$2"
shift 2

: "${SCIREPEVAL_REPO:?Set SCIREPEVAL_REPO to the cloned allenai/scirepeval repository.}"
: "${HF_DATASETS_CACHE:?Set HF_DATASETS_CACHE to the prepared Hugging Face datasets cache.}"
PYTHON_BIN="${SCIREPEVAL_PYTHON:-python}"
BATCH_SIZE="${SCIREPEVAL_BATCH_SIZE:-16}"
TASKS_CONFIG="${SCIREPEVAL_TASKS_CONFIG:-$SCIREPEVAL_REPO/scirepeval_tasks.jsonl}"

[[ -f "$SCIREPEVAL_REPO/scirepeval.py" ]] || { echo "Missing scirepeval.py: $SCIREPEVAL_REPO" >&2; exit 2; }
[[ -f "$TASKS_CONFIG" ]] || { echo "Missing task config: $TASKS_CONFIG" >&2; exit 2; }
[[ -d "$HF_DATASETS_CACHE" ]] || { echo "Missing HF datasets cache: $HF_DATASETS_CACHE" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }

# The original SciRepEval Model class treats every local checkpoint as a run
# directory containing model/ and tokenizer/. Standard BGE-M3 checkpoints keep
# both at their root, so present a symlinked compatibility wrapper when needed.
MODEL_FOR_SCIREPEVAL="$MODEL_INPUT"
if [[ -d "$MODEL_INPUT" ]]; then
  if [[ -f "$MODEL_INPUT/model/config.json" && -d "$MODEL_INPUT/tokenizer" ]]; then
    : # Already in SciRepEval's expected local-checkpoint layout.
  elif [[ -f "$MODEL_INPUT/config.json" ]]; then
    WRAPPER_DIR="$OUTPUT_DIR/.scirepeval_bge_m3_wrapper"
    mkdir -p "$WRAPPER_DIR"
    ln -sfn "$MODEL_INPUT" "$WRAPPER_DIR/model"
    ln -sfn "$MODEL_INPUT" "$WRAPPER_DIR/tokenizer"
    MODEL_FOR_SCIREPEVAL="$WRAPPER_DIR"
  else
    echo "Local model path has neither config.json nor model/config.json: $MODEL_INPUT" >&2
    exit 2
  fi
fi

mkdir -p "$OUTPUT_DIR"
export HF_DATASETS_CACHE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASK_ARGS=()
if (( $# > 0 )); then
  TASK_ARGS=(--task-list "$@")
fi
FP16_ARGS=()
if [[ "${SCIREPEVAL_FP16:-0}" == "1" ]]; then
  FP16_ARGS=(--fp16)
fi

cd "$SCIREPEVAL_REPO"
"$PYTHON_BIN" scirepeval.py \
  --mtype default \
  --model "$MODEL_FOR_SCIREPEVAL" \
  --pooling-mode cls \
  --batch-size "$BATCH_SIZE" \
  --tasks-config "$TASKS_CONFIG" \
  --output "$OUTPUT_DIR/scirepeval_results.json" \
  "${FP16_ARGS[@]}" \
  "${TASK_ARGS[@]}"
