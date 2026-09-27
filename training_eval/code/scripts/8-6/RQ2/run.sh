#!/bin/bash


# bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/RQ2/eval_beir.sh  \
#     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1/merged_model" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-beir/RQ2/msmarco-baseline/train-qwen3_0.6b-v1-nq" \
#     "nq"




bash /data/share/project/tr/science_benchmark/run_flagembedding_custom_eval.sh  \
    "birco_relic" \
    "/data/share/project/tr/science_benchmark/converted_flagembedding_custom/birco_relic" \
    "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1/merged_model" \
    "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-birco_relic-flag-embedding/RQ2/msmarco-baseline/train-qwen3_0.6b-v1"


# bash /data/share/project/tr/science_benchmark/run_flagembedding_custom_eval.sh  \
#     "evidencebench" \
#     "/data/share/project/tr/science_benchmark/converted_flagembedding_custom/evidencebench" \
#     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1/merged_model" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-evidencebench-flag-embedding/RQ2/msmarco-baseline/train-qwen3_0.6b-v1"


# cqadupstack_math_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/cqadupstack-mathematica/cqadupstack-mathematica-10k-20260805.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/cqadupstack-mathematica/cqadupstack-mathematica-10k-20260805_exclude_other_pos_save_non_pos.jsonl"
# )


# for idx in "${!cqadupstack_math_jsonl_files[@]}"; do
#     jsonl_file="${cqadupstack_math_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="cqadupstack-math"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/RQ2/train-qwen3_0.6b.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/8-6/RQ2/$task_name/train-qwen3_0.6b-merged" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name/train-qwen3_0.6b-merged.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/RQ2/eval_beir.sh  \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/8-6/RQ2/$task_name/train-qwen3_0.6b-merged/merged_model" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-beir/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged" \
    #     "cqadupstack"
   
# done



#
# nq_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/nq/nq-10k-1q-20260806_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/nq/nq-10k-1q-20260806.jsonl"
# )


# for idx in "${!nq_jsonl_files[@]}"; do
#     jsonl_file="${nq_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="nq"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/RQ2/train-qwen3_0.6b.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/8-6/RQ2/$task_name/train-qwen3_0.6b-merged" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name/train-qwen3_0.6b-merged.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/RQ2/eval_beir.sh  \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/8-6/RQ2/$task_name/train-qwen3_0.6b-merged/merged_model" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-beir/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged" \
    #     "$task_name"
   
# done







birco_relic_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/birco-relic/birco-relic-20260801-gen-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/birco-relic/birco-relic-20260801-gen-query.jsonl"
)


for idx in "${!birco_relic_jsonl_files[@]}"; do
    jsonl_file="${birco_relic_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="birco_relic"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/RQ2/train-qwen3_0.6b.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/8-6-birco_relic-flag-embedding/RQ2/$task_name/train-qwen3_0.6b-merged" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name/train-qwen3_0.6b-merged.log


    bash /data/share/project/tr/science_benchmark/run_flagembedding_custom_eval.sh  \
        "birco_relic" \
        "/data/share/project/tr/science_benchmark/converted_flagembedding_custom/birco_relic" \
        "/data/share/project/tr/data_synthesis_agent/model_pth/8-6-birco_relic-flag-embedding/RQ2/$task_name/train-qwen3_0.6b-merged/merged_model" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-birco_relic-flag-embedding/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged"

done


# evidencebench_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/evidencebench/evidencebench-10k-1q-20260806_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/evidencebench/evidencebench-10k-1q-20260806.jsonl"
# )


# for idx in "${!evidencebench_jsonl_files[@]}"; do
#     jsonl_file="${evidencebench_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="evidencebench"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/RQ2/train-qwen3_0.6b.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/8-6-evidencebench-flag-embedding/RQ2/$task_name/train-qwen3_0.6b-merged" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name/train-qwen3_0.6b-merged.log

#     bash /data/share/project/tr/science_benchmark/run_flagembedding_custom_eval.sh  \
#         "evidencebench" \
#         "/data/share/project/tr/science_benchmark/converted_flagembedding_custom/evidencebench" \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/8-6-evidencebench-flag-embedding/RQ2/$task_name/train-qwen3_0.6b-merged/merged_model" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-evidencebench-flag-embedding/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged"
   
# done