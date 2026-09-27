#!/bin/bash
set -euo pipefail



EVAL_SCRIPT="/data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/eval_beir.sh"
MODEL_ROOT="/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale"
OUTPUT_ROOT="/data/share2/project/tr/data_synthesis_agent/eval_output/7-3-beir-new/RQ3_scale"
LOG_ROOT="/data/share/project/tr/data_synthesis_agent/logs/7-3-new-beir/RQ3_scale"

# 10k/20k blocks are intentionally skipped here, matching the previous active
# script behavior. Add them to this list if they need to be re-evaluated.
tasks=(
    "msmarco-10k"
    "msmarco-20k"
    "msmarco-40k"
    "msmarco-80k"
    "msmarco-160k"
    "msmarco-320k"
)

for task_name in "${tasks[@]}"; do
    model_path="$MODEL_ROOT/$task_name/train-qwen3_8b-merged"
    save_dir="$OUTPUT_ROOT/$task_name/train-qwen3_8b-merged"
    log_dir="$LOG_ROOT/$task_name"
    log_path="$log_dir/eval_beir.log"

    if [ ! -d "$model_path" ]; then
        echo "Missing model path: $model_path" >&2
        exit 1
    fi

    mkdir -p "$log_dir" "$save_dir"
    echo "[$(date +%F_%T)] evaluating $task_name"
    echo "model_path=$model_path"
    echo "save_dir=$save_dir"

    bash "$EVAL_SCRIPT" "$model_path" "$save_dir" 2>&1 | tee "$log_path"
done
