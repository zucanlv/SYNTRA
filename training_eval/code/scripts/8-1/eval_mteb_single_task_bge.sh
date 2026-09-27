#!/bin/bash

# set huggingface mirror
export HF_ENDPOINT=https://hf-mirror.com
export MTEB_CACHE="/data/share/project/shared_datasets/mteb"

# set model cache dir
export HF_HUB_CACHE="/data/share/project/shared_models"
export HF_DATASETS_CACHE="/data/share/project/shared_datasets/mteb"

eval_root="/data/share/project/tr/mmteb/code/train/qwen3-8b"


model_path=$1
out_dir=$2
task=$3
echo "model_path: $model_path"
echo "out_dir: $out_dir"
echo "task: $task"
mkdir -p "$out_dir"



cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /data/share/project/tr/mmteb/code/train/qwen3-8b/code/main.py \
    --benchmark_name \"$task\" \
    --results_output_folder \"$out_dir\" \
    --model_name_or_path \"$model_path\" \
    --use_fp16 True \
    --use_bf16 False \
    --prompt_template '{}' \
    --assert_prompts_exist False \
    --normalize_embeddings True \
    --batch_size 4 \
    --max_length 512 \
    "

echo "$cmd"
eval "$cmd"