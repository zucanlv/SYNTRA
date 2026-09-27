#!/bin/bash

default_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-1-dt-selection/dt-select-10k.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-1-dt-selection/dt-select-10k_exclude_other_pos_save_non_pos.jsonl"
)

random_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-1-dt-selection/random-sample-10k.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-1-dt-selection/random-sample-10k_exclude_other_pos_save_non_pos.jsonl"
)
for idx in "${!default_jsonl_files[@]}"; do
    jsonl_file="${default_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/5-4/default

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-4/train-qwen3_0.6b-msmarco.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/5-4/default/train-qwen3_0.6b-msmarco_anno-v$realidx" \
    #     $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/5-4/default/train-qwen3_0.6b-msmarco_anno-v$realidx.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-4/eval_beir_qwen3_0.6b-msmarco.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/5-4/default/train-qwen3_0.6b-msmarco_anno-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/5-4-msmarco/default/train-qwen3_0.6b-msmarco-anno_v$realidx" 

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/eval_nano_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-4/default/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-4-nanobeir/default/train-qwen3_0.6b-msmarco-anno_v$realidx"

    python   /data/share/project/tr/mmteb/code/train/qwen3-8b/code/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-4-nanobeir/default/train-qwen3_0.6b-msmarco-anno_v$realidx/no_model_name_available/no_revision_available" \
         > "/data/share/project/tr/data_synthesis_agent/eval_output/5-4-nanobeir/default/train-qwen3_0.6b-msmarco-anno_v$realidx/summary.md"

done


for idx in "${!random_jsonl_files[@]}"; do
    jsonl_file="${random_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/5-4/random

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-4/train-qwen3_0.6b-msmarco.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/5-4/random/train-qwen3_0.6b-msmarco_anno-v$realidx" \
    #     $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/5-4/random/train-qwen3_0.6b-msmarco_anno-v$realidx.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-4/eval_beir_qwen3_0.6b-msmarco.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/5-4/random/train-qwen3_0.6b-msmarco_anno-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/5-4-msmarco/random/train-qwen3_0.6b-msmarco-anno_v$realidx" 

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/eval_nano_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-4/random/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-4-nanobeir/random/train-qwen3_0.6b-msmarco-anno_v$realidx"

    python   /data/share/project/tr/mmteb/code/train/qwen3-8b/code/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-4-nanobeir/random/train-qwen3_0.6b-msmarco-anno_v$realidx/no_model_name_available/no_revision_available" \
         > "/data/share/project/tr/data_synthesis_agent/eval_output/5-4-nanobeir/random/train-qwen3_0.6b-msmarco-anno_v$realidx/summary.md"

done
