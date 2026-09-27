#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export MTEB_CACHE="/data/share/project/shared_datasets/mteb"
export HF_HUB_CACHE="/data/share/project/shared_datasets/mteb"
DECODER_ONLY_MODELS=(
#   "nvidia/NV-Embed-v2"
#   "intfloat/e5-mistral-7b-instruct"
#   "Alibaba-NLP/gte-Qwen2-7B-instruct"
  # "BAAI/bge-multilingual-gemma2"
  # "NovaSearch/stella_en_1.5B_v5"
  # "jinaai/jina-embeddings-v5-text-small"
  # "Qwen/Qwen3-Embedding-8B"
  # "Qwen/Qwen3-Embedding-4B"
  # "Qwen/Qwen3-Embedding-0.6B"
  "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
)

ENCODER_ONLY_MODELS=(
#   "Snowflake/snowflake-arctic-embed-l"
#   "Snowflake/snowflake-arctic-embed-m"
#   "Snowflake/snowflake-arctic-embed-xs"
#   "BAAI/bge-large-en-v1.5"
#   "BAAI/bge-base-en-v1.5"
#   "BAAI/bge-small-en-v1.5"
#   "intfloat/e5-large-v2"
#   "intfloat/e5-base-v2"
#   "intfloat/e5-small-v2"
#   "Alibaba-NLP/gte-modernbert-base"
)


for model in "${DECODER_ONLY_MODELS[@]}"; do

  MODEL_ROOT="/data/share/project/shared_models"
  MODEL_ID=$model
  base_name=${MODEL_ID//\//__}
  LOCAL_MODEL_DIR="${MODEL_ROOT}/${MODEL_ID//\//__}"
  eval_output_folder="/data/share/project/tr/data_synthesis_agent/eval_output/5-7-nanobeir/decoder_only/$base_name"
  bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-7/nanobeir_beir_consistency/eval_nano_beir.sh \
    "$LOCAL_MODEL_DIR" \
    "$eval_output_folder"
  
  python /data/share/project/tr/mmteb/code/summary.py \
    "$eval_output_folder"/*/* \
    > "$eval_output_folder/summary.md"
done

for model in "${ENCODER_ONLY_MODELS[@]}"; do

  MODEL_ROOT="/data/share/project/shared_models"
  MODEL_ID=$model
  base_name=${MODEL_ID//\//__}
  LOCAL_MODEL_DIR="${MODEL_ROOT}/${MODEL_ID//\//__}"
  eval_output_folder="/data/share/project/tr/data_synthesis_agent/eval_output/5-7-nanobeir/encoder_only/$base_name"
  bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-7/nanobeir_beir_consistency/eval_nano_beir.sh \
    "$LOCAL_MODEL_DIR" \
    "$eval_output_folder"
  
  python /data/share/project/tr/mmteb/code/summary.py \
    "$eval_output_folder"/*/* \
    > "$eval_output_folder/summary.md"
done