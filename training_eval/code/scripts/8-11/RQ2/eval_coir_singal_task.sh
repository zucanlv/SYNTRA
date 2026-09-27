#!/bin/bash

# set huggingface mirror
export HF_ENDPOINT=https://hf-mirror.com
export MTEB_CACHE="/data/share/project/shared_datasets/mteb"

# set model cache dir
export HF_HUB_CACHE="/data/share/project/shared_datasets"
export HF_DATASETS_CACHE="/data/share/project/shared_datasets/mteb"

cache_dir=/data/share/project/tr/data/coir/cache


# 源文件路径列表
required_files=(
  "/data/share/project/shared_models/Qwen3-Embedding-0.6B/1_Pooling/config.json"
  "/data/share/project/shared_models/Qwen3-Embedding-0.6B/config_sentence_transformers.json"
  "/data/share/project/shared_models/Qwen3-Embedding-0.6B/modules.json"
  "/data/share/project/shared_models/Qwen3-Embedding-0.6B/sentence_bert_config.json"
)

# all_tasks=(
#   "codetrans-contest" 
#   "codetrans-dl" 
#   "cosqa" 
#   "synthetic-text2sql" 
#   "stackoverflow-qa" 
#   "codefeedback-mt" 
#   "codefeedback-st" 
#   "CodeSearchNet-ccr-go" 
#   "CodeSearchNet-ccr-java" 
#   "CodeSearchNet-ccr-javascript" 
#   "CodeSearchNet-ccr-php" 
#   "CodeSearchNet-ccr-python" 
#   "CodeSearchNet-ccr-ruby" 
#   "CodeSearchNet-go" 
#   "CodeSearchNet-java" 
#   "CodeSearchNet-javascript" 
#   "CodeSearchNet-php" 
#   "CodeSearchNet-python" 
#   "CodeSearchNet-ruby" 
# )

model_path=$1
save_dir=$2
task=$3
mkdir -p "$save_dir"
eval_script_root=/data/share/project/tr/third_party/FlagEmbedding/research/BGE_Coder/evaluation/coir_eval


cd "$eval_script_root"
cmd="python main.py \
    --use_special_instructions True \
    --output_dir \"$save_dir\" \
    --use_bf16 True \
    --use_fp16 False \
    --embedder_name_or_path  \"$model_path\" \
    --embedder_model_class decoder-only-base \
    --embedder_query_max_length 2048 \
    --embedder_passage_max_length 2048 \
    --trust_remote_code True \
    --pooling_method last_token \
    --embedder_batch_size 64 \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 \
    --query_instruction_format_for_retrieval 'Instruct: {}\nQuery: {}' \
    --tasks "$task" \
    --cache_dir $cache_dir
    "

echo "$cmd"
eval "$cmd"

# Example

# python main.py \
#     --output_dir ${output_dir} \
#     --use_special_instructions True \
#     --embedder_name_or_path BAAI/bge-code-v1 \
#     --embedder_model_class decoder-only-base \
#     --query_instruction_format_for_retrieval '<instruct>{}\n<query>{}' \
#     --embedder_query_max_length 2048 \
#     --embedder_passage_max_length 2048 \
#     --trust_remote_code True \
#     --pooling_method last_token \
#     --embedder_batch_size 64 \
#     --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
#     --tasks apps codetrans-contest codetrans-dl cosqa synthetic-text2sql stackoverflow-qa codefeedback-mt codefeedback-st CodeSearchNet-ccr-go CodeSearchNet-ccr-java CodeSearchNet-ccr-javascript CodeSearchNet-ccr-php CodeSearchNet-ccr-python CodeSearchNet-ccr-ruby CodeSearchNet-go CodeSearchNet-java CodeSearchNet-javascript CodeSearchNet-php CodeSearchNet-python CodeSearchNet-ruby \
#     --cache_dir ./cache