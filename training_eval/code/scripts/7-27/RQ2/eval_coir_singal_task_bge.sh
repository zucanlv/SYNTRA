#!/bin/bash

# set huggingface mirror
export HF_ENDPOINT=https://hf-mirror.com
export MTEB_CACHE="/data/share/project/shared_datasets/mteb"

# set model cache dir
export HF_HUB_CACHE="/data/share/project/shared_datasets"
export HF_DATASETS_CACHE="/data/share/project/shared_datasets/mteb"

cache_dir=/data/share/project/tr/data/coir/cache


model_path=$1
save_dir=$2
task=$3
mkdir -p "$save_dir"
eval_script_root=/data/share/project/tr/third_party/FlagEmbedding/research/BGE_Coder/evaluation/coir_eval


cd "$eval_script_root"
cmd="python main.py \
    --output_dir \"$save_dir\" \
    --use_bf16 False \
    --use_fp16 True \
    --embedder_name_or_path  \"$model_path\" \
    --embedder_model_class encoder-only-base \
    --embedder_query_max_length 2048 \
    --embedder_passage_max_length 2048 \
    --pooling_method cls \
    --embedder_batch_size 64 \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --tasks "$task" \
    --cache_dir $cache_dir
    "

echo "$cmd"
eval "$cmd"
