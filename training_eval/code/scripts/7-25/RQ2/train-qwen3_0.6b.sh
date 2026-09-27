#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE=disabled

export HF_HUB_CACHE="/data/share/project/shared_datasets/.cache"

MODEL_PATH="/data/share/project/shared_models/Qwen3-0.6B-auto-eos"

BASE_OUTPUT_DIR=$1
SYN_DATA=$2
mkdir -p $BASE_OUTPUT_DIR

num_train_epochs=1
per_device_train_batch_size=16
num_gpus=6

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5


#################### 通用训练函数 ####################
run_once() {   
    local train_data="$1" 

    local output_dir="${BASE_OUTPUT_DIR}"
    mkdir -p "${output_dir}"

    echo "============================================================"
    echo "Train data:"
    echo "  ${train_data}"
    echo "Output dir: ${output_dir}"
    echo "============================================================"

    torchrun --nproc_per_node "${num_gpus}" \
        --master_port 29501 \
        -m FlagEmbedding.finetune.embedder.decoder_only.base \
        --model_name_or_path "${MODEL_PATH}" \
        --cache_dir "${HF_HUB_CACHE}" \
        --use_lora True \
        --lora_rank 32 \
        --lora_alpha 64 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
        --save_merged_lora_model True \
        --use_mrl False \
        \
        --trust_remote_code True \
        --train_data ${train_data} \
        --cache_path "${HF_HUB_CACHE}" \
        --train_group_size 8 \
        --query_max_len 512 \
        --passage_max_len 512 \
        --pad_to_multiple_of 8 \
        --query_instruction_format "Instruct: {}\nQuery: {}" \
        --knowledge_distillation True \
        --same_dataset_within_batch True \
        --small_threshold 0 \
        --drop_threshold 0 \
        \
        --output_dir "${output_dir}" \
        --overwrite_output_dir \
        --learning_rate 1e-4 \
        --bf16 \
        --num_train_epochs "${num_train_epochs}" \
        --per_device_train_batch_size "${per_device_train_batch_size}" \
        --dataloader_drop_last True \
        --warmup_ratio 0.1 \
        --gradient_checkpointing \
        --deepspeed /data/share/project/public_envs/FlagEmbedding/examples/finetune/ds_stage1.json \
        --logging_steps 1 \
        --save_steps 1000 \
        --negatives_cross_device \
        --temperature 0.02 \
        --sentence_pooling_method last_token \
        --normalize_embeddings True \
        --kd_loss_type kl_div

}


run_once "${SYN_DATA}" 