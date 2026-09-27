#!/bin/bash

bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-25/RQ2/eval_coir.sh \
    "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1" \
    "/data/share/project/tr/data_synthesis_agent/eval_output/6-25-coir/RQ3_scale/msmarco-10k-merged/train-qwen3_0.6b-v1"

cosqa_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/cosqa-0618.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/cosqa-0618_exclude_other_pos_save_non_pos.jsonl"
)


for idx in "${!cosqa_jsonl_files[@]}"; do
    jsonl_file="${cosqa_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="cosqa"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-25/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-25/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-25/RQ2/$task_name/train-qwen3_0.6b-merged" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-25/RQ2/$task_name/train-qwen3_0.6b-merged.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-25/RQ2/eval_coir_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/6-25/RQ2/$task_name/train-qwen3_0.6b-merged" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-25-coir/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
        "$task_name"
   
done


cosqa_official_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/cosqa-official-0624.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/cosqa-official-0624_exclude_other_pos_save_non_pos.jsonl"
)

for idx in "${!cosqa_official_jsonl_files[@]}"; do
    jsonl_file="${cosqa_official_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="cosqa"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-25/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-25/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-25/RQ2/$task_name/train-qwen3_0.6b-merged-official" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-25/RQ2/$task_name/train-qwen3_0.6b-merged-official.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-25/RQ2/eval_coir_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/6-25/RQ2/$task_name/train-qwen3_0.6b-merged-official" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-25-coir/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
        "$task_name"
   
done