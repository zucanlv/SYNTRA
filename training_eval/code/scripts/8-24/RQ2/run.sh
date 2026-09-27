#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run.sh [bge-m3|qwen3-8b|all]
# Override SYN_DATA_DIR, CUDA_VISIBLE_DEVICES, NUM_TRAIN_EPOCHS, or batch-size
# environment variables when running on a different machine.

MODEL_SELECTION="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/data/share/project/tr"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
SYN_DATA_DIR="${SYN_DATA_DIR:-/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/scirepeval-search}"
OUTPUT_ROOT="$ROOT_DIR/data_synthesis_agent/model_pth/8-24/RQ2/scirepeval"
EVAL_ROOT="$ROOT_DIR/data_synthesis_agent/eval_output/8-24-scirepeval-full-corpus-corrected/RQ2/scirepeval"
LOG_ROOT="$ROOT_DIR/data_synthesis_agent/logs/8-24/RQ2/scirepeval"

SYNTHETIC_FILES=(
  "$SYN_DATA_DIR/scirepeval-search-10k-20260812_exclude_other_pos_save_non_pos.jsonl"
  "$SYN_DATA_DIR/scirepeval-search-10k-20260812.jsonl"
)

case "$MODEL_SELECTION" in
  bge-m3|qwen3-8b|all) ;;
  *) echo "Usage: $0 [bge-m3|qwen3-8b|all]" >&2; exit 2 ;;
esac
for data_file in "${SYNTHETIC_FILES[@]}"; do
  [[ -f "$data_file" ]] || { echo "Missing synthetic data: $data_file" >&2; exit 2; }
done

run_model() {
  local model_family="$1" train_script="$2" model_suffix="$3" model_subdir="$4"
  local idx=0 data_file train_dir model_dir eval_dir log_dir
  for data_file in "${SYNTHETIC_FILES[@]}"; do
    idx=$((idx + 1))
    train_dir="$OUTPUT_ROOT/$model_subdir/train-$model_suffix-v$idx"
    model_dir="$train_dir"
    [[ "$model_family" == qwen3-8b ]] && model_dir="$train_dir/merged_model"
    eval_dir="$EVAL_ROOT/$model_subdir/train-$model_suffix-v$idx"
    log_dir="$LOG_ROOT/$model_subdir"
    mkdir -p "$log_dir"

    bash "$train_script" "$train_dir" "$data_file" 2>&1 | tee "$log_dir/train-$model_suffix-v$idx.log"
    CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" \
      bash "$SCRIPT_DIR/eval_scirepeval_full_corpus.sh" "$model_dir" "$eval_dir" "$model_family" 10 \
      2>&1 | tee "$log_dir/eval-full-corpus-$model_suffix-v$idx.log"
  done
}

if [[ "$MODEL_SELECTION" == bge-m3 || "$MODEL_SELECTION" == all ]]; then
  run_model bge-m3 "$SCRIPT_DIR/train-bge_m3.sh" bge-m3 bge-m3
fi
if [[ "$MODEL_SELECTION" == qwen3-8b || "$MODEL_SELECTION" == all ]]; then
  run_model qwen3-8b "$SCRIPT_DIR/train-qwen3_8b.sh" qwen3-8b qwen3-8b
fi
