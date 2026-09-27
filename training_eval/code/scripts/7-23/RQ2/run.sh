#!/bin/bash

bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/eval_coir_singal_task.sh \
    "/data/share/project/shared_models/bge-m3" \
    "/data/share/project/tr/data_synthesis_agent/eval_output/7-23-coir/RQ2/bge-m3-baseline" \
    "cosqa"

bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/eval_coir_singal_task.sh \
    "/data/share/project/shared_models/bge-m3" \
    "/data/share/project/tr/data_synthesis_agent/eval_output/7-23-coir/RQ2/bge-m3-baseline" \
    "synthetic-text2sql"

cosqa_jsonl_files=(
    "/data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/processed_data/cosqa/cosqa-official-0624.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/processed_data/cosqa/cosqa-official-0624_exclude_other_pos_save_non_pos.jsonl"
    "/data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/processed_data/cosqa/cosqa-official-0624.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/processed_data/cosqa/cosqa-official-0624_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/processed_data/cosqa/cosqa-0618.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/processed_data/cosqa/cosqa-0618_exclude_other_pos_save_non_pos.jsonl"
)



for idx in "${!cosqa_jsonl_files[@]}"; do
    jsonl_file="${cosqa_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="cosqa"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-23/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/train-bge_m3.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/7-23/RQ2/$task_name/train-bge_m3-merged-$realidx" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-23/RQ2/$task_name/train-bge_m3-merged-$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/eval_coir_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/7-23/RQ2/$task_name/train-bge_m3-merged-$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/7-23-coir/RQ2/$task_name/train-bge_m3-merged-v$realidx" \
        "$task_name"
   
done




text2sql_jsonl_files=(
    "/data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/processed_data/text2sql/synthetic-text2sql-official-0627.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/processed_data/text2sql/synthetic-text2sql-official-0627_exclude_other_pos_save_non_pos.jsonl"
    "/data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/processed_data/text2sql/synthetic-text2sql-official-0627.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/processed_data/text2sql/synthetic-text2sql-official-0627_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/processed_data/text2sql/synthetic-text2sql-official-doc-gen-query-0627.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/processed_data/text2sql/synthetic-text2sql-official-doc-gen-query-0627_exclude_other_pos_save_non_pos.jsonl"
)

for idx in "${!text2sql_jsonl_files[@]}"; do
    jsonl_file="${text2sql_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="synthetic-text2sql"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-23/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/train-bge_m3.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/7-23/RQ2/$task_name/train-bge_m3-merged-$realidx" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-23/RQ2/$task_name/train-bge_m3-merged-$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-23/RQ2/eval_coir_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/7-23/RQ2/$task_name/train-bge_m3-merged-$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/7-23-coir/RQ2/$task_name/train-bge_m3-merged-v$realidx" \
        "$task_name"
   
done