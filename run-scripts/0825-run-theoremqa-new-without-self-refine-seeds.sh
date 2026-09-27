#!/usr/bin/env bash
set -euo pipefail

train_env=/home/ustc/public_envs/DSA
train_project=/data/share/project/tr/data_synthesis_agent
data_root=/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task
model_root="$train_project/model_pth/8-25/RQ2/theoremqa_theorems"
eval_root="$train_project/eval_output/8-25-bright/RQ2/theoremqa_theorems"
log_root="$train_project/logs/8-25/RQ2/theoremqa_theorems"
eval_script="$train_project/code/scripts/8-18/RQ2/eval_bright_singal_task.sh"

seeds=(42)
train_files=(
    "$data_root/bright-documents-theoremqa_theorems-10k-20260824-wo-self-refine.jsonl"
    "$data_root/bright-documents-theoremqa_theorems-10k-20260824-wo-self-refine_exclude_other_pos_save_non_pos.jsonl"
)

require_file() {
    local path=$1
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 2
    fi
}

gpus_are_idle() {
    local used

    if ps -eo cmd= | rg -q \
        '[t]orchrun|[F]lagEmbedding[.]finetune|[F]lagEmbedding[.]evaluation|python(3([.][0-9]+)?)? main[.]py .*--devices cuda'; then
        return 1
    fi

    mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ ${#used[@]} -ne 6 ]]; then
        return 1
    fi

    local value
    for value in "${used[@]}"; do
        if (( value >= 2048 )); then
            return 1
        fi
    done
}

wait_for_idle_gpus() {
    while true; do
        if gpus_are_idle; then
            sleep 30
            if gpus_are_idle; then
                return
            fi
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for all six GPUs to become idle ..."
        sleep 30
    done
}

for path in "${train_files[@]}" "$eval_script"; do
    require_file "$path"
done

mkdir -p "$model_root" "$eval_root" "$log_root"

export PATH="$train_env/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export WANDB_MODE=disabled
export HF_HUB_CACHE=/data/share/project/shared_datasets/.cache

for seed in "${seeds[@]}"; do
    run_name="train-qwen3_0.6b-without-self-refine-20260824-seed${seed}"
    model_dir="$model_root/$run_name"
    eval_dir="$eval_root/$run_name"
    train_log="$log_root/${run_name}-train.log"
    eval_log="$log_root/${run_name}-eval.log"

    if [[ -e "$model_dir" || -e "$eval_dir" ]]; then
        echo "Refusing to overwrite existing output for $run_name" >&2
        exit 3
    fi

    wait_for_idle_gpus
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training $run_name"
    "$train_env/bin/torchrun" \
        --nproc_per_node 6 \
        --master_port 29531 \
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

    wait_for_idle_gpus
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluating $run_name"
    bash "$eval_script" \
        "$model_dir" \
        "$eval_dir" \
        theoremqa_theorems \
        2>&1 | tee "$eval_log"

    result_file="$eval_dir/eval_results_examples.md"
    if [[ ! -s "$result_file" ]]; then
        echo "Missing evaluation result: $result_file" >&2
        exit 4
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed $run_name"
    sed -n '1,20p' "$result_file"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] New theoremqa without-self-refine seed run completed."
