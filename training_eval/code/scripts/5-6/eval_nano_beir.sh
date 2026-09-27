#!/bin/bash

# set huggingface mirror
export HF_ENDPOINT=https://hf-mirror.com
export MTEB_CACHE="/data/share/project/shared_datasets/mteb"

# set model cache dir
export HF_HUB_CACHE="/data/share/project/shared_datasets/mteb"

eval_root="/data/share/project/tr/mmteb/code/train/qwen3-8b"
model_path=/data/share/project/shared_models/Qwen3-Embedding-0.6B
results_output_folder=/data/share/project/tr/data_synthesis_agent/eval_output/5-6/nano_beir_qwen3_0_6b


# out_dir="$results_output_folder/geevec-qwen3-8b-v3-all-six-tasks-batchsize"
mkdir -p "$results_output_folder"

cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python /data/share/project/tr/mmteb/code/main.py \
    --benchmark_name \"NanoBEIR\" \
    --results_output_folder \"$results_output_folder\" \
    --model_name_or_path \"$model_path\" \
    --trust_remote_code 'True' \
    --use_fp16 False \
    --use_bf16 True \
    --prompt_template 'Instruct: {}\nQuery: ' \
    --assert_prompts_exist True \
    --normalize_embeddings True \
    --batch_size 16 \
    --max_length 512 \
    "

echo "$cmd"
eval "$cmd"
