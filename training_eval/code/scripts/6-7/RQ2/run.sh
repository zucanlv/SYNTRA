#!/bin/bash

touche2020_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0606-touche/touche-10k-0606.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0606-touche/touche-10k-0606_exclude_other_pos_save_non_pos.jsonl"
)

for idx in "${!touche2020_jsonl_files[@]}"; do
    jsonl_file="${touche2020_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="webis-touche2020"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-7/RQ2_general_task/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-7/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-7/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
        $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-7/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-7/RQ2/eval_beir_single_task.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-7/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-7-beir/RQ2_general_task/$task_name/train-qwen3_0.6b-_v$realidx" \
        "$task_name"


done
