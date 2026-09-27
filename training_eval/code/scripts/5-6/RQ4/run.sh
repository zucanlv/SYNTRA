#!/bin/bash

RQ4_Ablation_1_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-1-dt-selection/dt-select-10k_exclude_other_pos_save_non_pos.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-1-dt-selection/random-sample-10k_exclude_other_pos_save_non_pos.jsonl"
)

RQ4_Ablation_2_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-2-query-gen-instr/without-query-gen-instr-10k.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-2-query-gen-instr/without-query-gen-instr-10k_exclude_other_pos_save_non_pos.jsonl"
)

RQ4_Ablation_3_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-3-idattr/without-adattr-10k.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-3-idattr/without-adattr-10k_exclude_other_pos_save_non_pos.jsonl"
)
RQ4_Ablation_4_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-4-anno/hn-mine-10k.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-4-anno/hn-mine-10k-ori-pos-anno.jsonl"
)
RQ4_Ablation_5_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-5-self-refine/without-self-refine.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-5-self-refine/without-self-refine_exclude_other_pos_save_non_pos.jsonl"
)

# for idx in "${!RQ4_Ablation_1_jsonl_files[@]}"; do
#     jsonl_file="${RQ4_Ablation_1_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/5-6/RQ4_Ablation_1

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/train-qwen3_0.6b-msmarco.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_1/train-qwen3_0.6b-msmarco_anno-v$realidx" \
#         $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/5-6/RQ4_Ablation_1/train-qwen3_0.6b-msmarco_anno-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/eval_beir_qwen3_0.6b-msmarco.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_1/train-qwen3_0.6b-msmarco_anno-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-msmarco/RQ4_Ablation_1/train-qwen3_0.6b-msmarco-anno_v$realidx" 

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/eval_nano_beir.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_1/train-qwen3_0.6b-msmarco_anno-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_1/train-qwen3_0.6b-msmarco-anno_v$realidx"

#     python   /data/share/project/tr/mmteb/code/train/qwen3-8b/code/summary.py \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_1/train-qwen3_0.6b-msmarco-anno_v$realidx/no_model_name_available/no_revision_available" \
#          > "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_1/train-qwen3_0.6b-msmarco-anno_v$realidx/summary.md"


# done


for idx in "${!RQ4_Ablation_2_jsonl_files[@]}"; do
    jsonl_file="${RQ4_Ablation_2_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/5-6/RQ4_Ablation_2

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/train-qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_2/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/5-6/RQ4_Ablation_2/train-qwen3_0.6b-msmarco_anno-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/eval_beir_qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_2/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-msmarco/RQ4_Ablation_2/train-qwen3_0.6b-msmarco-anno_v$realidx" 

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/eval_nano_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_2/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_2/train-qwen3_0.6b-msmarco-anno_v$realidx"

    python   /data/share/project/tr/mmteb/code/train/qwen3-8b/code/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_2/train-qwen3_0.6b-msmarco-anno_v$realidx/no_model_name_available/no_revision_available" \
         > "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_2/train-qwen3_0.6b-msmarco-anno_v$realidx/summary.md"


done


for idx in "${!RQ4_Ablation_3_jsonl_files[@]}"; do
    jsonl_file="${RQ4_Ablation_3_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/5-6/RQ4_Ablation_3

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/train-qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_3/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/5-6/RQ4_Ablation_3/train-qwen3_0.6b-msmarco_anno-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/eval_beir_qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_3/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-msmarco/RQ4_Ablation_3/train-qwen3_0.6b-msmarco-anno_v$realidx" 

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/eval_nano_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_3/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_3/train-qwen3_0.6b-msmarco-anno_v$realidx"

    python   /data/share/project/tr/mmteb/code/train/qwen3-8b/code/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_3/train-qwen3_0.6b-msmarco-anno_v$realidx/no_model_name_available/no_revision_available" \
         > "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_3/train-qwen3_0.6b-msmarco-anno_v$realidx/summary.md"


done


for idx in "${!RQ4_Ablation_4_jsonl_files[@]}"; do
    jsonl_file="${RQ4_Ablation_4_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/5-6/RQ4_Ablation_4

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/train-qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_4/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/5-6/RQ4_Ablation_4/train-qwen3_0.6b-msmarco_anno-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/eval_beir_qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_4/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-msmarco/RQ4_Ablation_4/train-qwen3_0.6b-msmarco-anno_v$realidx" 

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/eval_nano_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_4/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_4/train-qwen3_0.6b-msmarco-anno_v$realidx"

    python   /data/share/project/tr/mmteb/code/train/qwen3-8b/code/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_4/train-qwen3_0.6b-msmarco-anno_v$realidx/no_model_name_available/no_revision_available" \
         > "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_4/train-qwen3_0.6b-msmarco-anno_v$realidx/summary.md"


done


for idx in "${!RQ4_Ablation_5_jsonl_files[@]}"; do
    jsonl_file="${RQ4_Ablation_5_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/5-6/RQ4_Ablation_5

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/train-qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_5/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        $jsonl_file 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/5-6/RQ4_Ablation_5/train-qwen3_0.6b-msmarco_anno-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/eval_beir_qwen3_0.6b-msmarco.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_5/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-msmarco/RQ4_Ablation_5/train-qwen3_0.6b-msmarco-anno_v$realidx" 

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/RQ4/eval_nano_beir.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/5-6/RQ4_Ablation_5/train-qwen3_0.6b-msmarco_anno-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_5/train-qwen3_0.6b-msmarco-anno_v$realidx"

    python   /data/share/project/tr/mmteb/code/train/qwen3-8b/code/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_5/train-qwen3_0.6b-msmarco-anno_v$realidx/no_model_name_available/no_revision_available" \
         > "/data/share/project/tr/data_synthesis_agent/eval_output/5-6-nanobeir/RQ4_Ablation_5/train-qwen3_0.6b-msmarco-anno_v$realidx/summary.md"


done