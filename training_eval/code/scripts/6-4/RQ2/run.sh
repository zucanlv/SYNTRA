#!/bin/bash

dbpedia_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0604/dbpedia-10k-0604-fewshot-cali-fill-easy-7.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0604/dbpedia-10k-0604-fewshot-cali-fill-easy-7_exclude_others_save_non_pos.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0604/dbpedia-0422-0604-10k.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0604/dbpedia-0422-0604-10k_exclude_others_save_non_pos.jsonl"
)

for idx in "${!dbpedia_jsonl_files[@]}"; do
    jsonl_file="${dbpedia_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="dbpedia-entity"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-4/RQ2_general_task/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-4/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-4/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
        $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-4/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-4/RQ2/eval_beir_single_task.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-4/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-4-beir/RQ2_general_task/$task_name/train-qwen3_0.6b-_v$realidx" \
        "$task_name"


done
