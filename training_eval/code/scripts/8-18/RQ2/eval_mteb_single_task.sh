#!/bin/bash

# set huggingface mirror
export HF_ENDPOINT=https://hf-mirror.com
export MTEB_CACHE="/data/share/project/shared_datasets/mteb"

# set model cache dir
export HF_HUB_CACHE="/data/share/project/shared_models"
export HF_DATASETS_CACHE="/data/share/project/shared_datasets/mteb"

eval_root="/data/share/project/tr/mmteb/code/train/qwen3-8b"


# 源文件路径列表
required_files=(
  "/data/share/project/shared_models/Qwen3-Embedding-0.6B/1_Pooling/config.json"
  "/data/share/project/shared_models/Qwen3-Embedding-0.6B/config_sentence_transformers.json"
  "/data/share/project/shared_models/Qwen3-Embedding-0.6B/modules.json"
  "/data/share/project/shared_models/Qwen3-Embedding-0.6B/sentence_bert_config.json"
)

model_path=$1
out_dir=$2
task=$3
echo "model_path: $model_path"
echo "out_dir: $out_dir"
echo "task: $task"
mkdir -p "$out_dir"



copy_required_files() {
  # for model_path in "${model_paths[@]}"; do
    # 遍历每个目标路径
    for required_file in "${required_files[@]}"; do
      # 提取文件名
      file_name=$(basename "$required_file")

      # 如果文件是 config.json 等，确保目标路径下有 1_Pooling 目录
      if [[ "$file_name" == "config.json" ]]; then
        target_dir="${model_path}/1_Pooling"
        
        # 如果 1_Pooling 子目录不存在，则创建
        if [[ ! -d "$target_dir" ]]; then
          echo "创建目录 $target_dir"
          mkdir -p "$target_dir"
        fi

        target_path="${target_dir}/${file_name}"
      else
        target_path="${model_path}/${file_name}"
      fi
      
      # 复制文件到目标路径
      echo "复制文件 $required_file 到 $target_path"
      cp "$required_file" "$target_path"
    done
  # done
}
copy_required_files 


cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python /data/share/project/tr/mmteb/code/train/qwen3-8b/code/main.py \
    --benchmark_name \"$task\" \
    --results_output_folder \"$out_dir\" \
    --model_name_or_path \"$model_path\" \
    --use_fp16 False \
    --use_bf16 True \
    --prompt_template 'Instruct: {}\nQuery: ' \
    --assert_prompts_exist False \
    --normalize_embeddings True \
    --batch_size 4 \
    --max_length 512 \
    "

echo "$cmd"
eval "$cmd"