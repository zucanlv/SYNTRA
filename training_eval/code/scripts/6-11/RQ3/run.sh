#!/bin/bash

msmarco_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori/0606/ori/msmarco-ori-10k-0611.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori/0606/ori/msmarco-ori-10k-0611_exclude_other_pos_save_non_pos.jsonl"
)


for idx in "${!msmarco_jsonl_files[@]}"; do
    jsonl_file="${msmarco_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-11/RQ3_scale/$task_name

    
    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-11/RQ3/train-qwen3_0.6b.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-11/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-11/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx.log


    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-11/RQ3/eval_nano_beir.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-11/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-11-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-11/RQ3/eval_beir_qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-11/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-11-msmarco/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"
    

    # python /data/share/project/tr/data_synthesis_agent/code/scripts/6-11/RQ3/summary.py \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-11-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"/*/* \
    #     > "/data/share/project/tr/data_synthesis_agent/eval_output/6-11-nanobeir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx/summary.md"

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-11/RQ3/eval_beir.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-11/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-11-beir/RQ3_scale/$task_name/train-qwen3_0.6b-v$realidx"
    
done
