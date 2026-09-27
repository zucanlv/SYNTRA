#!/bin/bash

msmarco10k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0627-gen/msmarco_0627_gen_10k.jsonl     /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0627-gen/msmarco_0627_gen_10k-exclude_other_pos_save_non_pos.jsonl"
)

msmarco20k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0627-gen/msmarco_0627_gen_20k.jsonl     /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0627-gen/msmarco_0627_gen_20k-exclude_other_pos_save_non_pos.jsonl"
)

msmarco40k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0627-gen/msmarco_0627_gen_40k.jsonl     /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0627-gen/msmarco_0627_gen_40k-exclude_other_pos_save_non_pos.jsonl"
)

msmarco80k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0627-gen/msmarco_0627_gen_80k.jsonl     /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0627-gen/msmarco_0627_gen_80k-exclude_other_pos_save_non_pos.jsonl"
)

msmarco160k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0627-gen/msmarco_0627_gen_160k.jsonl     /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0627-gen/msmarco_0627_gen_160k-exclude_other_pos_save_non_pos.jsonl"
)

msmarco320k_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0627-gen/msmarco_0627_gen_320k.jsonl     /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/0627-gen/msmarco_0627_gen_320k-exclude_other_pos_save_non_pos.jsonl"
)
for idx in "${!msmarco10k_jsonl_files[@]}"; do
    jsonl_file="${msmarco10k_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco-10k"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-27/RQ3_scale/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/train-qwen3_8b.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/eval_nano_beir.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged" "0" &

    python /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged"/*/* \
        > "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged/summary.md"
done


for idx in "${!msmarco20k_jsonl_files[@]}"; do
    jsonl_file="${msmarco20k_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco-20k"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-27/RQ3_scale/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/train-qwen3_8b.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/eval_nano_beir.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged" "1" &

    python /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged"/*/* \
        > "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged/summary.md"
done

for idx in "${!msmarco40k_jsonl_files[@]}"; do
    jsonl_file="${msmarco40k_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco-40k"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-27/RQ3_scale/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/train-qwen3_8b.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/eval_nano_beir.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged" "2" &

    python /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged"/*/* \
        > "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged/summary.md"
done

for idx in "${!msmarco80k_jsonl_files[@]}"; do
    jsonl_file="${msmarco80k_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco-80k"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-27/RQ3_scale/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/train-qwen3_8b.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/eval_nano_beir.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged" "3" &

    python /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged"/*/* \
        > "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged/summary.md"
done

for idx in "${!msmarco160k_jsonl_files[@]}"; do
    jsonl_file="${msmarco160k_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco-160k"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-27/RQ3_scale/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/train-qwen3_8b.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/eval_nano_beir.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged" "4" &

    python /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged"/*/* \
        > "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged/summary.md"
done

for idx in "${!msmarco320k_jsonl_files[@]}"; do
    jsonl_file="${msmarco320k_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco-320k"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-27/RQ3_scale/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/train-qwen3_8b.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/eval_nano_beir.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/6-27/RQ3_scale/$task_name/train-qwen3_8b-merged" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged" "5" &

    python /data/share/project/tr/data_synthesis_agent/code/scripts/6-27/RQ3/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged"/*/* \
        > "/data/share/project/tr/data_synthesis_agent/eval_output/6-27-nanobeir/RQ3_scale/$task_name/train-qwen3_8b-merged/summary.md"
done