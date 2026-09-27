#!/bin/bash

export HF_HUB_CACHE="/data/share/project/shared_datasets"
export HF_ENDPOINT=https://hf-mirror.com
if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_path="/data/share/project/shared_models/bge-m3"
save_dir="/data/share/project/tr/data_synthesis_agent/eval_output/7-2-beir-new-v2/bge-m3-correct"
mkdir -p "$save_dir"

cmd="python -m FlagEmbedding.evaluation.beir \
    --eval_name beir \
    --use_special_instructions False \
    --dataset_dir /data/share/project/tr/data/beir/eval/data \
    --dataset_names msmarco \
    --splits test dev \
    --corpus_embd_save_dir $save_dir/corpus_embd \
    --output_dir $save_dir/search_results \
    --search_top_k 1000 --rerank_top_k 100 \
    --cache_path $HF_HUB_CACHE \
    --overwrite False \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path $save_dir/beir_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
    --ignore_identical_ids True \
    --trust_remote_code True \
    --embedder_name_or_path $model_path \
    --embedder_model_class encoder-only-m3 \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --query_instruction_format_for_retrieval \"{}{}\" \
    --pooling_method cls \
    --use_fp16 True \
    --use_bf16 False \
    --embedder_batch_size 256 \
    --embedder_query_max_length 512 \
    --embedder_passage_max_length 512"

echo "$cmd"
eval "$cmd"
