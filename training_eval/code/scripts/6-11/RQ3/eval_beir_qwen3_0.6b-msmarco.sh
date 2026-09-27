#!/bin/bash

export HF_HUB_CACHE="/data/share/project/shared_datasets"
if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

# export HF_ENDPOINT="https://huggingface.co"
export HF_ENDPOINT=https://hf-mirror.com

# dataset_names="fiqa arguana cqadupstack"
# full dataset
# dataset_names="arguana climate-fever cqadupstack dbpedia-entity fever fiqa hotpotqa msmarco nfcorpus nq quora scidocs scifact trec-covid webis-touche2020"
# dataset_names="nq msmarco"
# dataset_names="fever fiqa hotpotqa nfcorpus nq quora scidocs scifact trec-covid webis-touche2020 msmarco"

model_path=$1
save_dir=$2
mkdir -p $save_dir

dataset_names="passage"

eval_args="\
    --eval_name msmarco \
    --dataset_names $dataset_names \
    --dataset_dir /data/share/project/tr/data/msmarco/eval/data \
    --splits dl19 dl20 \
    --corpus_embd_save_dir $save_dir/corpus_embd \
    --output_dir $save_dir/search_results \
    --search_top_k 1000 --rerank_top_k 100 \
    --cache_path $HF_HUB_CACHE \
    --overwrite False \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path $save_dir/msmarco_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
    --ignore_identical_ids True \
    --trust_remote_code True \
"

model_args="\
    --embedder_name_or_path  $model_path \
    --embedder_model_class decoder-only-base \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 \
    --query_instruction_format_for_retrieval 'Instruct: {}\nQuery: {}' \
    --pooling_method last_token \
    --use_fp16 True \
    --embedder_batch_size 128 \
    --embedder_query_max_length 512 \
    --embedder_passage_max_length 512 \
"

cmd="python -m FlagEmbedding.evaluation.msmarco \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd