#!/bin/bash
fiqa_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0614-fiqa/fiqa-10k-0614.jsonl"
    # "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0614-fiqa/fiqa-10k-0614_exclude_other_pos_save_non_pos.jsonl"
)

for idx in "${!fiqa_jsonl_files[@]}"; do
    jsonl_file="${fiqa_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="fiqa"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-16/RQ2_general_task/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-15/RQ2/train-qwen3_0.6b_figa_datamerge.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
        $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-15/RQ2/eval_beir_single_task.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-194" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-16-beir/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-194" \
        "$task_name"

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-15/RQ2/eval_beir_single_task.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-388" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-16-beir/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-388" \
        "$task_name"


done
