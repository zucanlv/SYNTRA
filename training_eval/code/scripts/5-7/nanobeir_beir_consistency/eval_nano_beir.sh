#!/bin/bash

# set huggingface mirror
export HF_ENDPOINT=https://hf-mirror.com
export MTEB_CACHE="/data/share/project/shared_datasets/mteb"

# set model cache dir
export HF_HUB_CACHE="/data/share/project/shared_datasets/mteb"

eval_root="/data/share/project/tr/mmteb/code/train/qwen3-8b"
model_path=$1
results_output_folder=$2

# required_files=(
#   "/data/share/project/shared_models/Qwen3-Embedding-0.6B/1_Pooling/config.json"
#   "/data/share/project/shared_models/Qwen3-Embedding-0.6B/config_sentence_transformers.json"
#   "/data/share/project/shared_models/Qwen3-Embedding-0.6B/modules.json"
#   "/data/share/project/shared_models/Qwen3-Embedding-0.6B/sentence_bert_config.json"
# )

# copy_required_files() {
#   # for model_path in "${model_paths[@]}"; do
#     # 遍历每个目标路径
#     for required_file in "${required_files[@]}"; do
#       # 提取文件名
#       file_name=$(basename "$required_file")

#       # 如果文件是 config.json 等，确保目标路径下有 1_Pooling 目录
#       if [[ "$file_name" == "config.json" ]]; then
#         target_dir="${model_path}/1_Pooling"
        
#         # 如果 1_Pooling 子目录不存在，则创建
#         if [[ ! -d "$target_dir" ]]; then
#           echo "创建目录 $target_dir"
#           mkdir -p "$target_dir"
#         fi

#         target_path="${target_dir}/${file_name}"
#       else
#         target_path="${model_path}/${file_name}"
#       fi
      
#       # 复制文件到目标路径
#       echo "复制文件 $required_file 到 $target_path"
#       cp "$required_file" "$target_path"
#     done
#   # done
# }
# copy_required_files 


# out_dir="$results_output_folder/geevec-qwen3-8b-v3-all-six-tasks-batchsize"
mkdir -p "$results_output_folder"

cmd="CUDA_VISIBLE_DEVICES=5 python /data/share/project/tr/mmteb/code/train/qwen3-8b/code/main.py \
    --benchmark_name \"NanoBEIR\" \
    --results_output_folder \"$results_output_folder\" \
    --model_name_or_path \"$model_path\" \
    --trust_remote_code 'True' \
    --use_fp16 False \
    --use_bf16 True \
    --prompt_template 'Instruct: {}\nQuery: ' \
    --assert_prompts_exist True \
    --normalize_embeddings True \
    --batch_size 1 \
    --max_length 512 \
    "

echo "$cmd"
eval "$cmd"
