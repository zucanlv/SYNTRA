#!/usr/bin/env bash
# Evaluate a Qwen3 embedding checkpoint with the original allenai/scirepeval Qwen3 adapter.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'USAGE'
Usage:
  SCIREPEVAL_REPO=/path/to/scirepeval \
  HF_DATASETS_CACHE=/path/to/huggingface/datasets/cache \
  SCIREPEVAL_PYTHON=/path/to/scirepeval_env/bin/python \
  bash eval_scirepeval_official_qwen3_0.6b.sh <qwen_model_path_or_hf_id> <output_dir> [task name ...]

Examples:
  # Run only the official Search task.
  ... eval_scirepeval_official_qwen3_0.6b.sh /models/merged_model outputs/qwen Search

  # Run every standard task in scirepeval_tasks.jsonl.
  ... eval_scirepeval_official_qwen3_0.6b.sh /models/merged_model outputs/qwen
USAGE
  exit 2
fi

MODEL_PATH="$1"
OUTPUT_DIR="$2"
shift 2

: "${SCIREPEVAL_REPO:?Set SCIREPEVAL_REPO to the cloned allenai/scirepeval repository.}"
: "${HF_DATASETS_CACHE:?Set HF_DATASETS_CACHE to the prepared Hugging Face datasets cache.}"
PYTHON_BIN="${SCIREPEVAL_PYTHON:-python}"
BATCH_SIZE="${SCIREPEVAL_BATCH_SIZE:-16}"
TASKS_CONFIG="${SCIREPEVAL_TASKS_CONFIG:-$SCIREPEVAL_REPO/scirepeval_tasks.jsonl}"
PROMPT_FILE="${SCIREPEVAL_PROMPT_FILE:-$SCIREPEVAL_REPO/instr_prompts.json}"
PROMPT_NAME="${SCIREPEVAL_PROMPT_NAME:-blank}"

[[ -f "$SCIREPEVAL_REPO/scirepeval.py" ]] || { echo "Missing scirepeval.py: $SCIREPEVAL_REPO" >&2; exit 2; }
[[ -f "$TASKS_CONFIG" ]] || { echo "Missing task config: $TASKS_CONFIG" >&2; exit 2; }
[[ -f "$PROMPT_FILE" ]] || { echo "Missing prompt JSON: $PROMPT_FILE" >&2; exit 2; }
[[ -d "$HF_DATASETS_CACHE" ]] || { echo "Missing HF datasets cache: $HF_DATASETS_CACHE" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
export HF_DATASETS_CACHE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASK_ARGS=()
if (( $# > 0 )); then
  TASK_ARGS=(--task-list "$@")
fi
TASK_PROMPT_ARGS=()
if [[ "${SCIREPEVAL_TASK_SPECIFIC_PROMPTS:-0}" == "1" ]]; then
  TASK_PROMPT_ARGS=(--task-specific-prompts)
fi

cd "$SCIREPEVAL_REPO"
"$PYTHON_BIN" scirepeval.py \
  --instructor \
  --model-type qwen3 \
  --model "$MODEL_PATH" \
  --prompt-file "$PROMPT_FILE" \
  --prompt-name "$PROMPT_NAME" \
  --batch-size "$BATCH_SIZE" \
  --tasks-config "$TASKS_CONFIG" \
  --output "$OUTPUT_DIR/scirepeval_results.json" \
  "${TASK_PROMPT_ARGS[@]}" \
  "${TASK_ARGS[@]}"
