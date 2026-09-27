#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE="/data/share/project/shared_datasets"
if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

# full datasets
# dataset_names="economics pony theoremqa_questions"

# small datasets for quick test
# dataset_names="pony theoremqa_theorems"


required_files=(
  "/data/share/project/shared_models/Qwen3-Embedding-8B/1_Pooling/config.json"
  "/data/share/project/shared_models/Qwen3-Embedding-8B/config_sentence_transformers.json"
  "/data/share/project/shared_models/Qwen3-Embedding-8B/modules.json"
  "/data/share/project/shared_models/Qwen3-Embedding-8B/sentence_bert_config.json"
)


model_path=$1
save_dir=$2
task=$3



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

model_args="\
    --embedder_name_or_path $model_path \
    --embedder_model_class decoder-only-base \
    --query_instruction_format_for_retrieval 'Instruct: {}\nQuery: {}' \
    --pooling_method last_token \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --cache_dir $HF_HUB_CACHE \
    --embedder_batch_size 16 \
    --embedder_query_max_length 8192 \
    --embedder_passage_max_length 8192 \
"

# split_list=("examples" "gpt4_reason")
split_list=("examples" )

for split in "${split_list[@]}"; do
    eval_args="\
        --task_type short \
        --use_special_instructions True \
        --eval_name bright_short \
        --dataset_dir /data/share2/project/tr/data/bright/data \
        --dataset_names "$task" \
        --splits $split \
        --corpus_embd_save_dir $save_dir/corpus_embd \
        --output_dir $save_dir/search_results/$split \
        --search_top_k 2000 \
        --cache_path $HF_HUB_CACHE \
        --overwrite False \
        --k_values 1 10 100 \
        --eval_output_method markdown \
        --eval_output_path $save_dir/eval_results_$split.md \
        --eval_metrics ndcg_at_10 recall_at_10 recall_at_100 \
    "

    cmd="python -m FlagEmbedding.evaluation.bright \
        $eval_args \
        $model_args \
    "

    echo $cmd
    eval $cmd

done