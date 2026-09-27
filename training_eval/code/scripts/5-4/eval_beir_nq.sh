#!/bin/bash

export HF_HUB_CACHE="/data/share/project/shared_datasets"
if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

export HF_ENDPOINT="https://huggingface.co"

# dataset_names="fiqa arguana cqadupstack"
# full dataset
dataset_names="nq"
# dataset_names="nq msmarco"
# dataset_names="arguana climate-fever cqadupstack dbpedia-entity fever fiqa hotpotqa nfcorpus quora scidocs scifact trec-covid webis-touche2020"
# dataset_names="nq dbpedia-entity fiqa nfcorpus scifact trec-covid webis-touche2020"
# dataset_names="arguana climate-fever cqadupstack fever hotpotqa  quora scidocs"

model_path=$1
save_dir=$2
mkdir -p $save_dir

eval_args="\
    --eval_name beir \
    --use_special_instructions True \
    --dataset_dir /data/share/project/tr/data/beir/eval/data \
    --dataset_names $dataset_names \
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
"

model_args="\
    --embedder_name_or_path  "$model_path" \
    --embedder_model_class decoder-only-base \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 \
    --query_instruction_format_for_retrieval 'Instruct: {}\nQuery: {}' \
    --pooling_method last_token \
    --use_bf16 True \
    --use_fp16 False \
    --embedder_batch_size 128 \
    --embedder_query_max_length 512 \
    --embedder_passage_max_length 512 \
"

cmd="python -m FlagEmbedding.evaluation.beir \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd