#!/bin/bash

theoremqa_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/theoremqa-theorem/deepseek-v4-pro-20260727_160957_exclude_other_pos_save_non_pos.jsonl       /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/theoremqa-theorem/deepseek-v4-pro-20260727_160957.jsonl"
)

for idx in "${!theoremqa_jsonl_files[@]}"; do
    jsonl_file="${theoremqa_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="theoremqa_theorems"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-28/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-28/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/7-28/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-28/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-28/RQ2/eval_bright_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/7-28/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/7-28-bright/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
        "$task_name"
   
done