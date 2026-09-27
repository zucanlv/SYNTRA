#!/bin/bash
cosqa_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/cosqa-official-doc-gen-query-0627.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/cosqa-official-doc-gen-query-0627_exclude_other_pos_save_non_pos.jsonl"
)


for idx in "${!cosqa_jsonl_files[@]}"; do
    jsonl_file="${cosqa_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="cosqa"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-28/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-28/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-28/RQ2/$task_name/train-qwen3_0.6b-merged" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-28/RQ2/$task_name/train-qwen3_0.6b-merged.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-28/RQ2/eval_coir_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/6-28/RQ2/$task_name/train-qwen3_0.6b-merged" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-28-coir/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
        "$task_name"
   
done


text2sql_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/synthetic-text2sql-official-doc-gen-query-0627.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/synthetic-text2sql-official-doc-gen-query-0627_exclude_other_pos_save_non_pos.jsonl"
)

for idx in "${!text2sql_jsonl_files[@]}"; do
    jsonl_file="${text2sql_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="synthetic-text2sql"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-28/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-28/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-28/RQ2/$task_name/train-qwen3_0.6b-merged" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-28/RQ2/$task_name/train-qwen3_0.6b-merged.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-28/RQ2/eval_coir_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/6-28/RQ2/$task_name/train-qwen3_0.6b-merged" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-28-coir/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
        "$task_name"
   
done



text2sql_official_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/synthetic-text2sql-official-0627.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/synthetic-text2sql-official-0627_exclude_other_pos_save_non_pos.jsonl"
)

for idx in "${!text2sql_official_jsonl_files[@]}"; do
    jsonl_file="${text2sql_official_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="synthetic-text2sql"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-28/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-28/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-28/RQ2/$task_name/train-qwen3_0.6b-merged-official" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-28/RQ2/$task_name/train-qwen3_0.6b-merged-official.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-28/RQ2/eval_coir_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/6-28/RQ2/$task_name/train-qwen3_0.6b-merged-official" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-28-coir/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
        "$task_name"
   
done