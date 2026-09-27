#!/bin/bash
msmarco_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/msmarco/deepseek-v4-flash.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/msmarco/deepseek-v4-flash_exclude_other_pos_save_non_pos.jsonl"
)


for idx in "${!msmarco_jsonl_files[@]}"; do
    jsonl_file="${msmarco_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-20/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-20/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/7-20/RQ2/$task_name/train-qwen3_0.6b-merged" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-20/RQ2/$task_name/train-qwen3_0.6b-merged.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-20/RQ2/eval_beir.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/7-20/RQ2/$task_name/train-qwen3_0.6b-merged" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/7-20-beir/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
        "$task_name"
   
done




text2sql_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/synthetic-text2sql/deepseek-v4-flash.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/synthetic-text2sql/deepseek-v4-flash_exclude_other_pos_save_non_pos.jsonl"
)

for idx in "${!text2sql_jsonl_files[@]}"; do
    jsonl_file="${text2sql_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="synthetic-text2sql"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-20/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-20/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/7-20/RQ2/$task_name/train-qwen3_0.6b-merged" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-20/RQ2/$task_name/train-qwen3_0.6b-merged.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-20/RQ2/eval_coir_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/7-20/RQ2/$task_name/train-qwen3_0.6b-merged" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/7-20-coir/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
        "$task_name"
   
done