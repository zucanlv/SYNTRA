#!/bin/bash

msmarco10k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0618-ori/msmarco-ori-10k-0618.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0618-ori/msmarco-ori-10k-0618_exclude_other_pos_save_non_pos.jsonl"
)

msmarco20k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0618-ori/msmarco-ori-20k-0618.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0618-ori/msmarco-ori-20k-0618_exclude_other_pos_save_non_pos.jsonl"
)

msmarco40k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0618-ori/msmarco-ori-40k-0618.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0618-ori/msmarco-ori-40k-0618_exclude_other_pos_save_non_pos.jsonl"
)

msmarco80k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0618-ori/msmarco-ori-80k-0618.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0618-ori/msmarco-ori-80k-0618_exclude_other_pos_save_non_pos.jsonl"
)

for idx in "${!msmarco10k_jsonl_files[@]}"; do
    jsonl_file="${msmarco10k_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco-10k"
    # mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-20/RQ3_scale/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/train-qwen3_0.6b.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/eval_nano_beir.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"

    # python /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/summary.py \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"/*/* \
    #     > "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/summary.md"
    
    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/eval_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-beir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"
   
done



for idx in "${!msmarco20k_jsonl_files[@]}"; do
    jsonl_file="${msmarco20k_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco-20k"
    # mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-20/RQ3_scale/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/train-qwen3_0.6b.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/eval_nano_beir.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"

    # python /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/summary.py \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"/*/* \
    #     > "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/summary.md"
    
    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/eval_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-beir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"

done



for idx in "${!msmarco40k_jsonl_files[@]}"; do
    jsonl_file="${msmarco40k_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco-40k"
    # mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-20/RQ3_scale/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/train-qwen3_0.6b.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/eval_nano_beir.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"

    # python /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/summary.py \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"/*/* \
    #     > "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/summary.md"
    
    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/eval_beir.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-beir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"
   
done


for idx in "${!msmarco80k_jsonl_files[@]}"; do
    jsonl_file="${msmarco80k_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco-80k"
    # mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-20/RQ3_scale/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/train-qwen3_0.6b.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/eval_nano_beir.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"

    # python /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/summary.py \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"/*/* \
    #     > "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/summary.md"
    
    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-20/RQ3/eval_beir.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-20-beir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"
   
done