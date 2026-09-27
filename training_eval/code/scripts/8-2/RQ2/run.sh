#!/bin/bash

# bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-2/RQ2/eval_beir.sh \
#     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1/merged_model" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-2-mteb/RQ2/msmarco-baseline/train-qwen3_0.6b-v1-cqadupstack" \
#     "cqadupstack"


# bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-2/RQ2/eval_mteb_single_task.sh \
#     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1/merged_model" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-2-mteb/RQ2/msmarco-baseline/train-qwen3_0.6b-v1-MedicalRetrieval" \
#     "MedicalRetrieval"


# bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-2/RQ2/eval_mteb_single_task.sh \
#     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1/merged_model" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-2-mteb/RQ2/msmarco-baseline/train-qwen3_0.6b-v1-MedicalRetrieval" \
#     "MedicalRetrieval"


# cqdupstack_stats_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/cqadupstack-stats/cqadupstack-stats-10k-0728.jsonl     /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/cqadupstack-stats/cqadupstack-stats-10k-0728_exclude_other_pos_save_non_pos.jsonl"
# )


# for idx in "${!cqdupstack_stats_jsonl_files[@]}"; do
#     jsonl_file="${cqdupstack_stats_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="cqadupstack-stats"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-2/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-2/RQ2/train-qwen3_0.6b.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/8-2/RQ2/$task_name/train-qwen3_0.6b-merged" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-2/RQ2/$task_name/train-qwen3_0.6b-merged.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-2/RQ2/eval_beir.sh  \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/8-2/RQ2/$task_name/train-qwen3_0.6b-merged/merged_model" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-2-mteb/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged" \
#         "cqadupstack"
   
# done



# #
# cqadupstack_physics_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/cqadupstack-physics/cqadupstack-physics-10k-0729_exclude_other_pos_save_non_pos.jsonl     /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/cqadupstack-physics/cqadupstack-physics-10k-0729.jsonl"
# )


# for idx in "${!cqadupstack_physics_jsonl_files[@]}"; do
#     jsonl_file="${cqadupstack_physics_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="cqadupstack-physics"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-2/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-2/RQ2/train-qwen3_0.6b.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/8-2/RQ2/$task_name/train-qwen3_0.6b-merged" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-2/RQ2/$task_name/train-qwen3_0.6b-merged.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-2/RQ2/eval_beir.sh  \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/8-2/RQ2/$task_name/train-qwen3_0.6b-merged/merged_model" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-2-mteb/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged" \
#         "cqadupstack"
   
# done






# MedicalRetrieval_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medicalretrieval/medicalretrieval-10k-0730_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medicalretrieval/medicalretrieval-10k-0730.jsonl"
# )


# for idx in "${!MedicalRetrieval_jsonl_files[@]}"; do
#     jsonl_file="${MedicalRetrieval_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="MedicalRetrieval"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-2/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-2/RQ2/train-qwen3_0.6b.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/8-2/RQ2/$task_name/train-qwen3_0.6b-merged" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-2/RQ2/$task_name/train-qwen3_0.6b-merged.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-2/RQ2/eval_mteb_single_task.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/8-2/RQ2/$task_name/train-qwen3_0.6b-merged/merged_model" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-2-mteb/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged" \
#         "$task_name"
   
# done




bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-2/RQ2/eval_beir_1024.sh \
    "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1/merged_model" \
    "/data/share/project/tr/data_synthesis_agent/eval_output/8-2-mteb-debug/RQ2/msmarco-baseline/train-qwen3_0.6b-v1-cqadupstack" \
    "cqadupstack"


cqadupstack_physics_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/cqadupstack-physics/cqadupstack-physics-10k-0729_exclude_other_pos_save_non_pos.jsonl     /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/cqadupstack-physics/cqadupstack-physics-10k-0729.jsonl"
)


for idx in "${!cqadupstack_physics_jsonl_files[@]}"; do
    jsonl_file="${cqadupstack_physics_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="cqadupstack-physics"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-2/RQ2/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-2/RQ2/train-qwen3_0.6b_1024.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/8-2-debug/RQ2/$task_name/train-qwen3_0.6b-merged" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-2-debug/RQ2/$task_name/train-qwen3_0.6b-merged.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-2/RQ2/eval_beir_1024.sh  \
       "/data/share/project/tr/data_synthesis_agent/model_pth/8-2-debug/RQ2/$task_name/train-qwen3_0.6b-merged/merged_model" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/8-2-mteb-debug/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged" \
        "cqadupstack"
   
done
