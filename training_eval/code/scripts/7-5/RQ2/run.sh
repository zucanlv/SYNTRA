#!/bin/bash


economics_ori_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task/bright-documents-economics-20260704-reason-embed-query.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task/bright-documents-economics-20260704-reason-embed-query_exclude_other_pos_save_non_pos.jsonl"
)


for idx in "${!economics_ori_jsonl_files[@]}"; do
    jsonl_file="${economics_ori_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="economics"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-5/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-5/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/7-5/RQ2/$task_name/train-qwen3_0.6b-merged-ori" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-5/RQ2/$task_name/train-qwen3_0.6b-merged-ori.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-5/RQ2/eval_bright_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/7-5/RQ2/$task_name/train-qwen3_0.6b-merged-ori" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/7-5-bright/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged-ori" \
        "$task_name"
   
done




economics_gen_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task/bright-documents-economics-20260704-gen-query.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task/bright-documents-economics-20260704-gen-query_exclude_other_pos_save_non_pos.jsonl"
)


for idx in "${!economics_gen_jsonl_files[@]}"; do
    jsonl_file="${economics_gen_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="economics"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-5/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-5/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/7-5/RQ2/$task_name/train-qwen3_0.6b-merged-gen" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-5/RQ2/$task_name/train-qwen3_0.6b-merged-gen.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-5/RQ2/eval_bright_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/7-5/RQ2/$task_name/train-qwen3_0.6b-merged-gen" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/7-5-bright/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged-gen" \
        "$task_name"
   
done

