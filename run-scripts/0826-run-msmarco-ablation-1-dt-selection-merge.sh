#!/usr/bin/env bash
set -euo pipefail

train_env="${TRAIN_ENV:-/home/ustc/public_envs/DSA}"
train_project="${TRAIN_PROJECT:-/data/share/project/tr/data_synthesis_agent}"
data_root="${DATA_ROOT:-/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-1-dt-selection}"

experiment="${EXPERIMENT:-ablation-1-dt-selection-merge}"
model_root="${MODEL_ROOT:-${train_project}/model_pth/8-25/RQ4-ablation-merge}"
log_root="${LOG_ROOT:-${train_project}/logs/8-25/RQ4-ablation-merge}"
msmarco_eval_root="${MSMARCO_EVAL_ROOT:-${train_project}/eval_output/8-25-msmarco/RQ4-ablation-merge}"
nanobeir_eval_root="${NANOBEIR_EVAL_ROOT:-${train_project}/eval_output/8-25-nanobeir/RQ4-ablation-merge}"

msmarco_eval_script="${MSMARCO_EVAL_SCRIPT:-${train_project}/code/scripts/6-16/RQ2/eval_beir_qwen3_0.6b-msmarco.sh}"
nanobeir_eval_script="${NANOBEIR_EVAL_SCRIPT:-${train_project}/code/scripts/6-16/RQ2/eval_nano_beir.sh}"
nanobeir_summary_script="${NANOBEIR_SUMMARY_SCRIPT:-/data/share/project/tr/mmteb/code/train/qwen3-8b/code/summary.py}"

raw_file="${data_root}/dt-select-10k.jsonl"
filtered_file="${data_root}/dt-select-10k_exclude_other_pos_save_non_pos.jsonl"
train_files=("$raw_file" "$filtered_file")

dry_run="${DRY_RUN:-0}"
skip_existing="${SKIP_EXISTING:-0}"
master_port="${MASTER_PORT:-29531}"
seed="${SEED:-42}"

model_dir="${model_root}/${experiment}"
msmarco_eval_dir="${msmarco_eval_root}/${experiment}"
nanobeir_eval_dir="${nanobeir_eval_root}/${experiment}"
train_log="${log_root}/${experiment}-train.log"
msmarco_eval_log="${log_root}/${experiment}-eval-msmarco.log"
nanobeir_eval_log="${log_root}/${experiment}-eval-nanobeir.log"

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
        '[t]orchrun|[F]lagEmbedding[.]finetune|[F]lagEmbedding[.]evaluation|python(3([.][0-9]+)?)? main[.]py .*--benchmark_name'; then
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

for path in "${train_files[@]}"; do
    require_file "$path"
done
require_file "$msmarco_eval_script"
require_file "$nanobeir_eval_script"
require_file "$nanobeir_summary_script"

if [[ "$dry_run" == "1" ]]; then
    echo "[DRY RUN] train ${experiment}: ${train_files[*]} -> ${model_dir}"
    echo "[DRY RUN] eval-msmarco ${experiment}: ${model_dir} -> ${msmarco_eval_dir}"
    echo "[DRY RUN] eval-nanobeir ${experiment}: ${model_dir} -> ${nanobeir_eval_dir}"
    exit 0
fi

if [[ ! -x "${train_env}/bin/torchrun" ]]; then
    echo "Missing torchrun: ${train_env}/bin/torchrun" >&2
    exit 2
fi

if [[ "$skip_existing" != "1" ]] && \
    { [[ -e "$model_dir" ]] || [[ -e "$msmarco_eval_dir" ]] || [[ -e "$nanobeir_eval_dir" ]]; }; then
    echo "Refusing to overwrite existing output for ${experiment}" >&2
    echo "Set SKIP_EXISTING=1 to resume completed stages." >&2
    exit 3
fi

mkdir -p "$model_root" "$log_root" "$msmarco_eval_root" "$nanobeir_eval_root"

export PATH="${train_env}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export WANDB_MODE=disabled
export HF_HUB_CACHE=/data/share/project/shared_datasets/.cache

if [[ "$skip_existing" == "1" && -s "${model_dir}/merged_model/model.safetensors" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping completed training: ${experiment}"
else
    wait_for_idle_gpus
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training ${experiment}"
    echo "Train data: ${train_files[*]}"
    "${train_env}/bin/torchrun" \
        --nproc_per_node 6 \
        --master_port "$master_port" \
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
fi

require_file "${model_dir}/merged_model/model.safetensors"

if [[ "$skip_existing" == "1" && -s "${msmarco_eval_dir}/msmarco_eval_results.md" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping completed MS MARCO evaluation: ${experiment}"
else
    wait_for_idle_gpus
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluating MS MARCO: ${experiment}"
    bash "$msmarco_eval_script" "$model_dir" "$msmarco_eval_dir" \
        2>&1 | tee "$msmarco_eval_log"
    require_file "${msmarco_eval_dir}/msmarco_eval_results.md"
fi

if [[ "$skip_existing" == "1" && -s "${nanobeir_eval_dir}/summary.md" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping completed NanoBEIR evaluation: ${experiment}"
else
    wait_for_idle_gpus
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluating NanoBEIR: ${experiment}"
    bash "$nanobeir_eval_script" "$model_dir" "$nanobeir_eval_dir" \
        2>&1 | tee "$nanobeir_eval_log"
    nanobeir_result_dir="${nanobeir_eval_dir}/no_model_name_available/no_revision_available"
    require_file "${nanobeir_result_dir}/NanoMSMARCORetrieval.json"
    python "$nanobeir_summary_script" "$nanobeir_result_dir" \
        > "${nanobeir_eval_dir}/summary.md"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ${experiment}"
sed -n '1,20p' "${msmarco_eval_dir}/msmarco_eval_results.md"
sed -n '1,40p' "${nanobeir_eval_dir}/summary.md"
