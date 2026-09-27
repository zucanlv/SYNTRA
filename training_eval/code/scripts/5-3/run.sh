#!/bin/bash

ori_exclude_other_pos_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/title-msmarco-10k/title-msmarco-74k-ori-v1-10k_exclude_other_pos_save_non_pos.jsonl"
)

for idx in "${!ori_exclude_other_pos_jsonl_files[@]}"; do
    jsonl_file="${ori_exclude_other_pos_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/5-3/gen_exclude_other_pos

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-3/train-qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-3/gen_exclude_other_pos/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/5-3/gen_exclude_other_pos/train-qwen3_0.6b-msmarco_anno-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-3/eval_beir_qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-3/gen_exclude_other_pos/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-3-msmarco/gen_exclude_other_pos/train-qwen3_0.6b-msmarco-anno_v$realidx" 

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-3/eval_beir_nq.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-3/gen_exclude_other_pos/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-3-beir/gen_exclude_other_pos/train-qwen3_0.6b-msmarco-anno_v$realidx"


done

