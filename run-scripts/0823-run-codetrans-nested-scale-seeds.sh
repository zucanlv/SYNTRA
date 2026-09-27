#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 {nested-1q|nested-2q|full-4q} SEED" >&2
    exit 2
fi

variant=$1
seed=$2

train_env=/data/share/project/public_envs/embedder_train_eval
project_root=/data/share/project/tr/data_synthesis_agent
data_root=/data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/codetrans-dl
model_root="$project_root/model_pth/8-23/RQ2/codetrans-dl"
eval_root="$project_root/eval_output/8-23-coir/RQ2/codetrans-dl"
log_root="$project_root/logs/8-23/RQ2/codetrans-dl"
eval_script_root="$project_root/../third_party/FlagEmbedding/research/BGE_Coder/evaluation/coir_eval"

query1_files=(
    "$data_root/codetrans-dl-4q-query1-20260812_exclude_other_pos_save_non_pos.jsonl"
    "$data_root/codetrans-dl-4q-query1-20260812.jsonl"
)
query2_files=(
    "$data_root/codetrans-dl-4q-query2-20260812_exclude_other_pos_save_non_pos.jsonl"
    "$data_root/codetrans-dl-4q-query2-20260812.jsonl"
)
query3_files=(
    "$data_root/codetrans-dl-4q-query3-20260812_exclude_other_pos_save_non_pos.jsonl"
    "$data_root/codetrans-dl-4q-query3-20260812.jsonl"
)
query4_files=(
    "$data_root/codetrans-dl-4q-query4-20260812_exclude_other_pos_save_non_pos.jsonl"
    "$data_root/codetrans-dl-4q-query4-20260812.jsonl"
)

case "$variant" in
    nested-1q)
        train_files=("${query1_files[@]}")
        ;;
    nested-2q)
        train_files=("${query1_files[@]}" "${query2_files[@]}")
        ;;
    full-4q)
        train_files=(
            "${query1_files[@]}"
            "${query2_files[@]}"
            "${query3_files[@]}"
            "${query4_files[@]}"
        )
        ;;
    *)
        echo "Unknown variant: $variant" >&2
        exit 2
        ;;
esac

run_name="train-qwen3_0.6b-${variant}-strict-order-seed${seed}"
model_dir="$model_root/$run_name"
eval_dir="$eval_root/$run_name"
train_log="$log_root/${run_name}-train.log"
eval_log="$log_root/${run_name}-eval.log"

if [[ -e "$model_dir" || -e "$eval_dir" ]]; then
    echo "Refusing to overwrite existing output for $run_name" >&2
    exit 3
fi

mkdir -p "$model_root" "$eval_root" "$log_root"

export PATH="$train_env/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE=/data/share/project/shared_datasets/.cache
export HF_DATASETS_CACHE=/data/share/project/shared_datasets/mteb

echo "Training $run_name with ${#train_files[@]} files"
"$train_env/bin/torchrun" \
    --nproc_per_node 6 \
    --master_port 29523 \
    -m FlagEmbedding.finetune.embedder.decoder_only.base \
    --model_name_or_path /data/share/project/shared_models/Qwen3-0.6B-auto-eos \
    --cache_dir /data/share/project/shared_datasets/.cache \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --save_merged_lora_model True \
    --use_mrl False \
    --trust_remote_code True \
    --train_data "${train_files[@]}" \
    --cache_path /data/share/project/shared_datasets/.cache \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_format $'Instruct: {}\nQuery: {}' \
    --knowledge_distillation True \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --output_dir "$model_dir" \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
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
    --kd_loss_type kl_div \
    --seed "$seed" \
    2>&1 | tee "$train_log"

echo "Evaluating $run_name"
(
    cd "$eval_script_root"
    "$train_env/bin/python" main.py \
        --use_special_instructions True \
        --output_dir "$eval_dir" \
        --use_bf16 True \
        --use_fp16 False \
        --embedder_name_or_path "$model_dir/merged_model" \
        --embedder_model_class decoder-only-base \
        --embedder_query_max_length 2048 \
        --embedder_passage_max_length 2048 \
        --trust_remote_code True \
        --pooling_method last_token \
        --embedder_batch_size 64 \
        --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 \
        --query_instruction_format_for_retrieval $'Instruct: {}\nQuery: {}' \
        --tasks codetrans-dl \
        --cache_dir /data/share/project/tr/data/coir/cache
) 2>&1 | tee "$eval_log"

result_file="$eval_dir/merged_model/OVERALL-results.json"
if [[ ! -s "$result_file" ]]; then
    echo "Missing evaluation result: $result_file" >&2
    exit 4
fi

echo "Completed $run_name"
cat "$result_file"
