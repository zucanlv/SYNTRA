#!/bin/bash


bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-18/RQ2/eval_bright_singal_task.sh \
    "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/msmarco/train-qwen3_0.6b-v1" \
    "/data/share/project/tr/data_synthesis_agent/eval_output/8-18-bright/RQ2/theoremqa_theorems/msmarco-baseline/train-qwen3_0.6b-v1" \
    "theoremqa_theorems"


bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-18/RQ2/eval_beir.sh \
    "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/msmarco/train-qwen3_0.6b-v1" \
    "/data/share/project/tr/data_synthesis_agent/eval_output/8-18-beir/RQ2/arguana/msmarco-baseline/train-qwen3_0.6b-v1" \
    "arguana"

theoremqa_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task/bright-documents-theoremqa_theorems-10k-20260816-wo-self-refine.jsonl     /data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task/bright-documents-theoremqa_theorems-10k-20260816-wo-self-refine_exclude_other_pos_save_non_pos.jsonl"
)

for idx in "${!theoremqa_jsonl_files[@]}"; do
    jsonl_file="${theoremqa_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="theoremqa_theorems"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-18/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-18/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/8-18/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-18/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-18/RQ2/eval_bright_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/8-18/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/8-18-bright/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
        "$task_name"
   
done


arguana_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/arguana/arguana-20260816-wo-self-refine.jsonl     /data/share/project/shared_datasets/DSA/syn_data/arguana/arguana-20260816-wo-self-refine_exclude_other_pos_save_non_pos.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/arguana/arguana-20260816-with-self-refine.jsonl /data/share/project/shared_datasets/DSA/syn_data/arguana/arguana-20260816-with-self-refine_exclude_other_pos_save_non_pos.jsonl"
)

for idx in "${!arguana_jsonl_files[@]}"; do
    jsonl_file="${arguana_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="arguana"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-18/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-18/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/8-18/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-18/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-18/RQ2/eval_beir.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/8-18/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/8-18-beir/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
        "$task_name"
   
done