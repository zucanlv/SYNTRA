#!/bin/bash

msmarco10k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/msmarco-gen-nested-10k.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/msmarco-gen-nested_exclude_other_pos_save_non_pos-10k.jsonl"
)

msmarco20k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/msmarco-gen-nested-20k.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/msmarco-gen-nested_exclude_other_pos_save_non_pos-20k.jsonl"
)

msmarco40k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/msmarco-gen-nested-40k.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/msmarco-gen-nested_exclude_other_pos_save_non_pos-40k.jsonl"
)
msmarco80k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/msmarco-gen-nested-80k.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/msmarco-gen-nested_exclude_other_pos_save_non_pos-80k.jsonl"
)
msmarco160k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/msmarco-gen-nested-160k.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/msmarco-gen-nested_exclude_other_pos_save_non_pos-160k.jsonl"
)
msmarco320k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/msmarco-gen-nested-320k.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/msmarco-gen-nested_exclude_other_pos_save_non_pos-320k.jsonl"
)


for idx in "${!msmarco160k_jsonl_files[@]}"; do
    jsonl_file="${msmarco160k_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco-160k"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/5-11/RQ3_scale/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-11/RQ3/train-qwen3_0.6b.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/5-11/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/5-11/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-11/RQ3/eval_nano_beir.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/5-11/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/5-11-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"

    # python /data/share/project/tr/mmteb/code/summary.py \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/5-11-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"/*/* \
    #     > "/data/share/project/tr/data_synthesis_agent/eval_output/5-11-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/summary.md"
    
    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-11/RQ3/eval_beir_qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-11/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-1000" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-13-msmarco/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx-checkpoint-1000"

done
