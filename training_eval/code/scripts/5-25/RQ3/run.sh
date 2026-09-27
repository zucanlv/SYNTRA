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
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/5-25/RQ3_scale/$task_name

    
    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-25/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
        $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/5-25/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx.log

    # ckpt-100
    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/eval_nano_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-25/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-100" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-100"

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/eval_beir_qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-25/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-100" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-msmarco/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-100"
    

    python /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"/checkpoint-100/*/* \
        > "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-100/summary.md"

    # ckpt-200

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/eval_nano_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-25/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-200" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-200"

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/eval_beir_qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-25/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-200" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-msmarco/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-200"
    

    python /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"/checkpoint-200/*/* \
        > "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-200/summary.md"
    
    # ckpt-400
    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/eval_nano_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-25/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-400" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-400"

    python /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"/checkpoint-400/*/* \
        > "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-400/summary.md"

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/eval_beir_qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-25/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-400" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-msmarco/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-400"
    
    # ckpt-800
    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/eval_nano_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-25/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-800" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-800"

    python /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"/checkpoint-800/*/* \
        > "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-800/summary.md"
    
    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/eval_beir_qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-25/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-800" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-msmarco/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-800"

    # ckpt-1600
    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/eval_nano_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-25/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-1600" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-1600"

    python /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"/checkpoint-1600/*/* \
        > "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-1600/summary.md"
    
    
    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-25/RQ3/eval_beir_qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-25/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-1600" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-25-msmarco/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/checkpoint-1600"

done
