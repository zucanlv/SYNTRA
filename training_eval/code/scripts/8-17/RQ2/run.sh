#!/usr/bin/env bash
set -euo pipefail

# Official SciRepEval Qwen3 environment and locally prepared official Search data.
export PATH="/data/share/project/public_envs/embedder_train_eval/bin:$PATH"
SCIREPEVAL_REPO="/data/share/project/tr/third_party/scirepeval"
SCIREPEVAL_PYTHON="$SCIREPEVAL_REPO/.venv-qwen3/bin/python"
HF_DATASETS_CACHE="/data/share/project/shared_datasets/.cache/hf_datasets"
SCIREPEVAL_TASKS_CONFIG="$SCIREPEVAL_REPO/local_data/search/tasks_search_local.jsonl"
# All requested Search files are local; do not make accidental Hub requests at evaluation time.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

[[ -f "$SCIREPEVAL_REPO/scirepeval.py" ]] || { echo "Missing SciRepEval checkout: $SCIREPEVAL_REPO" >&2; exit 2; }
[[ -x "$SCIREPEVAL_PYTHON" ]] || { echo "Missing SciRepEval Python: $SCIREPEVAL_PYTHON" >&2; exit 2; }
[[ -d "$HF_DATASETS_CACHE" ]] || { echo "Missing datasets cache: $HF_DATASETS_CACHE" >&2; exit 2; }
[[ -f "$SCIREPEVAL_TASKS_CONFIG" ]] || { echo "Missing local Search task config: $SCIREPEVAL_TASKS_CONFIG" >&2; exit 2; }

scirepeval_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/scirepeval-search/scirepeval-search-10k-20260812_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/scirepeval-search/scirepeval-search-10k-20260812.jsonl"
)

for idx in "${!scirepeval_jsonl_files[@]}"; do
    jsonl_file="${scirepeval_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="scirepeval"
    train_dir="/data/share/project/tr/data_synthesis_agent/model_pth/8-17/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx"
    eval_dir="/data/share/project/tr/data_synthesis_agent/eval_output/8-17-scirepeval-official/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged"
    log_dir="/data/share/project/tr/data_synthesis_agent/logs/8-17/RQ2/$task_name"
    mkdir -p "$log_dir"

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-17/RQ2/train-qwen3_0.6b.sh \
    #     "$train_dir" \
    #     "$jsonl_file" 2>&1 | tee "$log_dir/train-qwen3_0.6b-$realidx-merged.log"

    # SCIREPEVAL_REPO="$SCIREPEVAL_REPO" \
    # SCIREPEVAL_PYTHON="$SCIREPEVAL_PYTHON" \
    # SCIREPEVAL_TASKS_CONFIG="$SCIREPEVAL_TASKS_CONFIG" \
    # HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-17/RQ2/eval_scirepeval_official_qwen3_0.6b.sh \
    #     "$train_dir/merged_model" \
    #     "$eval_dir" \
    #     "Search" 2>&1 | tee "$log_dir/eval-scirepeval-official-$realidx.log"


    SCIREPEVAL_REPO="$SCIREPEVAL_REPO" \
    SCIREPEVAL_PYTHON="$SCIREPEVAL_PYTHON" \
    SCIREPEVAL_TASKS_CONFIG="$SCIREPEVAL_TASKS_CONFIG" \
    HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-17/RQ2/eval_scirepeval_official_qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/msmarco/train-qwen3_0.6b-v1" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/8-17-scirepeval-official/RQ2/msmarco-baseline/$task_name/train-qwen3_0.6b-v$realidx-merged" \
        "Search" 2>&1 | tee "$log_dir/eval-scirepeval-official-$realidx.log"
done
