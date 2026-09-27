#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {lr5e5-e1|lr3e5-e1|lr1e4-ms22|reverse-lr5e5-e1|mixed-batch-lr1e4-e1|continue2q-all4-lr1e5-ms8|continue2q-all4-lr1e5-ms16|continue2q-all4-lr5e6-ms16|continue2q-weighted4q-lr1e6-ms2|continue2q-weighted4q-lr1e6-ms3|continue2q-weighted4q-lr1e6-ms8|continue2q-weighted4q-lr2e6-ms8|continue2q-weighted4q-lr1e6-ms16|continue2q-tail4q-lr1e6-ms3|independent2q-plus-tail200|independent2q-plus-unfiltered-tail|probe-q2|probe-q3|probe-q4}" >&2
    exit 2
fi

config=$1
train_env=/data/share/project/public_envs/embedder_train_eval
project_root=/data/share/project/tr/data_synthesis_agent
data_root=/data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/codetrans-dl
model_root="$project_root/model_pth/8-23/RQ2/codetrans-dl"
eval_root="$project_root/eval_output/8-23-coir/RQ2/codetrans-dl"
log_root="$project_root/logs/8-23/RQ2/codetrans-dl"
eval_script_root="$project_root/../third_party/FlagEmbedding/research/BGE_Coder/evaluation/coir_eval"
weighted_mix="$project_root/train_data_aliases/8-23/codetrans-dl/filtered-q1q2-w10-q3q4-w1-seed20260823.jsonl"
mix_head_weight=10
mix_tail_weight=1
mix_max_per_query=0
target_score=0.37961
train_gpus=${TRAIN_GPUS:-0,1,2,3,4,5}
train_seed=${TRAIN_SEED:-42}
run_suffix=${RUN_SUFFIX:-}
train_gpus_without_commas=${train_gpus//,/}
nproc=$((1 + ${#train_gpus} - ${#train_gpus_without_commas}))
eval_devices=()
for ((device_index = 0; device_index < nproc; device_index++)); do
    eval_devices+=("cuda:$device_index")
done

q1=(
    "$data_root/codetrans-dl-4q-query1-20260812.jsonl"
    "$data_root/codetrans-dl-4q-query1-20260812_exclude_other_pos_save_non_pos.jsonl"
)
q2=(
    "$data_root/codetrans-dl-4q-query2-20260812.jsonl"
    "$data_root/codetrans-dl-4q-query2-20260812_exclude_other_pos_save_non_pos.jsonl"
)
q3=(
    "$data_root/codetrans-dl-4q-query3-20260812.jsonl"
    "$data_root/codetrans-dl-4q-query3-20260812_exclude_other_pos_save_non_pos.jsonl"
)
q4=(
    "$data_root/codetrans-dl-4q-query4-20260812.jsonl"
    "$data_root/codetrans-dl-4q-query4-20260812_exclude_other_pos_save_non_pos.jsonl"
)

independent_2q=(
    "$data_root/codetrans-dl-2q-query2-20260811.jsonl"
    "$data_root/codetrans-dl-2q-query2-20260811_exclude_other_pos_save_non_pos.jsonl"
    "$data_root/codetrans-dl-2q-query1-20260811.jsonl"
    "$data_root/codetrans-dl-2q-query1-20260811_exclude_other_pos_save_non_pos.jsonl"
)

learning_rate=
save_steps=
schedule_args=()
base_model=/data/share/project/shared_models/Qwen3-0.6B-auto-eos
train_files=("${q1[@]}" "${q2[@]}" "${q3[@]}" "${q4[@]}")
run_label="full-4q-$config"
same_dataset_within_batch=True
knowledge_distillation=True
build_weighted_mix=False

case "$config" in
    lr5e5-e1)
        learning_rate=5e-5
        save_steps=10
        schedule_args=(--num_train_epochs 1 --warmup_ratio 0.1)
        ;;
    lr3e5-e1)
        learning_rate=3e-5
        save_steps=10
        schedule_args=(--num_train_epochs 1 --warmup_ratio 0.1)
        ;;
    lr1e4-ms22)
        learning_rate=1e-4
        save_steps=11
        schedule_args=(--max_steps 22 --warmup_steps 2)
        ;;
    reverse-lr5e5-e1)
        learning_rate=5e-5
        save_steps=10
        schedule_args=(--num_train_epochs 1 --warmup_ratio 0.1)
        train_files=("${q4[@]}" "${q3[@]}" "${q2[@]}" "${q1[@]}")
        ;;
    mixed-batch-lr1e4-e1)
        learning_rate=1e-4
        save_steps=1000
        schedule_args=(--num_train_epochs 1 --warmup_ratio 0.1)
        same_dataset_within_batch=False
        ;;
    continue2q-all4-lr1e5-ms8)
        learning_rate=1e-5
        save_steps=1000
        schedule_args=(--max_steps 8 --warmup_steps 0)
        base_model=/data/share/project/tr/data_synthesis_agent/model_pth/8-11/RQ2/codetrans-dl/train-qwen3_0.6b-merged-v2/merged_model
        run_label=full-4q-from-best2q-lr1e5-ms8
        ;;
    continue2q-all4-lr1e5-ms16)
        learning_rate=1e-5
        save_steps=1000
        schedule_args=(--max_steps 16 --warmup_steps 0)
        base_model=/data/share/project/tr/data_synthesis_agent/model_pth/8-11/RQ2/codetrans-dl/train-qwen3_0.6b-merged-v2/merged_model
        run_label=full-4q-from-best2q-lr1e5-ms16
        ;;
    continue2q-all4-lr5e6-ms16)
        learning_rate=5e-6
        save_steps=1000
        schedule_args=(--max_steps 16 --warmup_steps 0)
        base_model=/data/share/project/tr/data_synthesis_agent/model_pth/8-11/RQ2/codetrans-dl/train-qwen3_0.6b-merged-v2/merged_model
        run_label=full-4q-from-best2q-lr5e6-ms16
        ;;
    continue2q-weighted4q-lr1e6-ms8)
        learning_rate=1e-6
        save_steps=1000
        schedule_args=(--max_steps 8 --warmup_steps 0)
        base_model=/data/share/project/tr/data_synthesis_agent/model_pth/8-11/RQ2/codetrans-dl/train-qwen3_0.6b-merged-v2/merged_model
        train_files=("$weighted_mix")
        knowledge_distillation=False
        build_weighted_mix=True
        run_label=weighted-4q-from-best2q-lr1e6-ms8
        ;;
    continue2q-weighted4q-lr1e6-ms2)
        learning_rate=1e-6
        save_steps=1000
        schedule_args=(--max_steps 2 --warmup_steps 0)
        base_model=/data/share/project/tr/data_synthesis_agent/model_pth/8-11/RQ2/codetrans-dl/train-qwen3_0.6b-merged-v2/merged_model
        train_files=("$weighted_mix")
        knowledge_distillation=False
        build_weighted_mix=True
        run_label=weighted-4q-from-best2q-lr1e6-ms2
        ;;
    continue2q-weighted4q-lr1e6-ms3)
        learning_rate=1e-6
        save_steps=1000
        schedule_args=(--max_steps 3 --warmup_steps 0)
        base_model=/data/share/project/tr/data_synthesis_agent/model_pth/8-11/RQ2/codetrans-dl/train-qwen3_0.6b-merged-v2/merged_model
        train_files=("$weighted_mix")
        knowledge_distillation=False
        build_weighted_mix=True
        run_label=weighted-4q-from-best2q-lr1e6-ms3
        ;;
    continue2q-tail4q-lr1e6-ms3)
        learning_rate=1e-6
        save_steps=1000
        schedule_args=(--max_steps 3 --warmup_steps 0)
        base_model=/data/share/project/tr/data_synthesis_agent/model_pth/8-11/RQ2/codetrans-dl/train-qwen3_0.6b-merged-v2/merged_model
        weighted_mix="$project_root/train_data_aliases/8-23/codetrans-dl/filtered-q3q4-w1-seed20260823.jsonl"
        mix_head_weight=0
        mix_tail_weight=1
        train_files=("$weighted_mix")
        knowledge_distillation=False
        build_weighted_mix=True
        run_label=tail-4q-from-best2q-lr1e6-ms3
        ;;
    independent2q-plus-tail200)
        learning_rate=1e-4
        save_steps=1000
        schedule_args=(--num_train_epochs 1 --warmup_ratio 0.1)
        weighted_mix="$project_root/train_data_aliases/8-23/codetrans-dl/filtered-q3q4-100each-seed20260823.jsonl"
        mix_head_weight=0
        mix_tail_weight=1
        mix_max_per_query=100
        train_files=("${independent_2q[@]}" "$weighted_mix")
        build_weighted_mix=True
        run_label=independent2q-plus-filtered-q3q4-100each
        ;;
    independent2q-plus-unfiltered-tail)
        learning_rate=1e-4
        save_steps=1000
        schedule_args=(--num_train_epochs 1 --warmup_ratio 0.1)
        train_files=("${independent_2q[@]}" "${q3[0]}" "${q4[0]}")
        run_label=independent2q-plus-unfiltered-q3q4
        ;;
    continue2q-weighted4q-lr2e6-ms8)
        learning_rate=2e-6
        save_steps=1000
        schedule_args=(--max_steps 8 --warmup_steps 0)
        base_model=/data/share/project/tr/data_synthesis_agent/model_pth/8-11/RQ2/codetrans-dl/train-qwen3_0.6b-merged-v2/merged_model
        train_files=("$weighted_mix")
        knowledge_distillation=False
        build_weighted_mix=True
        run_label=weighted-4q-from-best2q-lr2e6-ms8
        ;;
    continue2q-weighted4q-lr1e6-ms16)
        learning_rate=1e-6
        save_steps=1000
        schedule_args=(--max_steps 16 --warmup_steps 0)
        base_model=/data/share/project/tr/data_synthesis_agent/model_pth/8-11/RQ2/codetrans-dl/train-qwen3_0.6b-merged-v2/merged_model
        train_files=("$weighted_mix")
        knowledge_distillation=False
        build_weighted_mix=True
        run_label=weighted-4q-from-best2q-lr1e6-ms16
        ;;
    probe-q2)
        learning_rate=1e-4
        save_steps=1000
        schedule_args=(--num_train_epochs 1 --warmup_ratio 0.1)
        train_files=("${q2[@]}")
        run_label=probe-q2
        ;;
    probe-q3)
        learning_rate=1e-4
        save_steps=1000
        schedule_args=(--num_train_epochs 1 --warmup_ratio 0.1)
        train_files=("${q3[@]}")
        run_label=probe-q3
        ;;
    probe-q4)
        learning_rate=1e-4
        save_steps=1000
        schedule_args=(--num_train_epochs 1 --warmup_ratio 0.1)
        train_files=("${q4[@]}")
        run_label=probe-q4
        ;;
    *)
        echo "Unknown config: $config" >&2
        exit 2
        ;;
esac

if [[ "$build_weighted_mix" == True ]] && [[ ! -s "$weighted_mix" ]]; then
    "$train_env/bin/python" data_scripts/build_codetrans_4q_weighted_mix.py \
        --data-root "$data_root" \
        --output "$weighted_mix" \
        --head-weight "$mix_head_weight" \
        --tail-weight "$mix_tail_weight" \
        --max-per-query "$mix_max_per_query" \
        --seed 20260823
fi

run_name="train-qwen3_0.6b-${run_label}-seed${train_seed}${run_suffix}"
model_dir="$model_root/$run_name"
train_log="$log_root/${run_name}-train.log"

if [[ -e "$model_dir" ]]; then
    echo "Refusing to overwrite existing model output: $model_dir" >&2
    exit 3
fi

mkdir -p "$model_root" "$eval_root" "$log_root"

export PATH="$train_env/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$train_gpus"
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE=/data/share/project/shared_datasets/.cache
export HF_DATASETS_CACHE=/data/share/project/shared_datasets/mteb

wait_for_gpus() {
    local selected=",$train_gpus,"
    while true; do
        local busy=0
        while IFS=',' read -r gpu_index memory utilization; do
            gpu_index=${gpu_index//[[:space:]]/}
            memory=${memory//[[:space:]]/}
            utilization=${utilization//[[:space:]]/}
            if [[ "$selected" == *",$gpu_index,"* ]] && (( memory > 1000 || utilization > 20 )); then
                busy=1
            fi
        done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

        if (( busy == 0 )); then
            return
        fi

        echo "[$(date -u '+%F %T')] Waiting for GPUs $train_gpus to become idle"
        sleep 30
    done
}

evaluate_model() {
    local model_path=$1
    local eval_name=$2
    local eval_dir="$eval_root/$eval_name"
    local eval_log="$log_root/${eval_name}-eval.log"
    local result_file="$eval_dir/merged_model/OVERALL-results.json"

    if [[ -s "$result_file" ]]; then
        echo "Reusing evaluation: $result_file"
    else
        echo "Evaluating $eval_name"
        (
            cd "$eval_script_root"
            "$train_env/bin/python" main.py \
                --use_special_instructions True \
                --output_dir "$eval_dir" \
                --use_bf16 True \
                --use_fp16 False \
                --embedder_name_or_path "$model_path" \
                --embedder_model_class decoder-only-base \
                --embedder_query_max_length 2048 \
                --embedder_passage_max_length 2048 \
                --trust_remote_code True \
                --pooling_method last_token \
                --embedder_batch_size 64 \
                --devices "${eval_devices[@]}" \
                --query_instruction_format_for_retrieval $'Instruct: {}\nQuery: {}' \
                --tasks codetrans-dl \
                --cache_dir /data/share/project/tr/data/coir/cache
        ) 2>&1 | tee "$eval_log"
    fi

    if [[ ! -s "$result_file" ]]; then
        echo "Missing evaluation result: $result_file" >&2
        exit 4
    fi

    local score
    score=$("$train_env/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["codetrans-dl"])' "$result_file")
    echo "RESULT $eval_name $score"
    if awk -v score="$score" -v target="$target_score" 'BEGIN { exit !(score > target) }'; then
        echo "TARGET_BEATEN $eval_name $score > $target_score"
    fi
}

wait_for_gpus

echo "Training $run_name with lr=$learning_rate, GPUs=$train_gpus, and ${#train_files[@]} files"
"$train_env/bin/torchrun" \
    --nproc_per_node "$nproc" \
    --master_port 29533 \
    -m FlagEmbedding.finetune.embedder.decoder_only.base \
    --model_name_or_path "$base_model" \
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
    --knowledge_distillation "$knowledge_distillation" \
    --same_dataset_within_batch "$same_dataset_within_batch" \
    --small_threshold 0 \
    --drop_threshold 0 \
    --output_dir "$model_dir" \
    --overwrite_output_dir \
    --learning_rate "$learning_rate" \
    --bf16 \
    "${schedule_args[@]}" \
    --per_device_train_batch_size 16 \
    --dataloader_drop_last True \
    --gradient_checkpointing \
    --deepspeed /data/share/project/public_envs/FlagEmbedding/examples/finetune/ds_stage1.json \
    --logging_steps 1 \
    --save_steps "$save_steps" \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
    --seed "$train_seed" \
    2>&1 | tee "$train_log"

while IFS= read -r checkpoint_dir; do
    step=${checkpoint_dir##*-}
    if (( step >= 20 )) && [[ -s "$checkpoint_dir/merged_model/model.safetensors" ]]; then
        evaluate_model "$checkpoint_dir/merged_model" "${run_name}-checkpoint${step}"
    fi
done < <(find "$model_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

evaluate_model "$model_dir/merged_model" "$run_name"
echo "Completed $run_name"
