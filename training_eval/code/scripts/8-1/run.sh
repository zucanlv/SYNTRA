#!/bin/bash


# bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/eval_mteb_single_task.sh \
#     "/data/share/project/shared_models/Qwen3-Embedding-8B" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-1-mteb/RQ2/qwen3-embedding-8b-baseline" \
#     "SpartQA"

# bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/eval_mteb_single_task.sh \
#     "/data/share/project/shared_models/Qwen3-Embedding-8B" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-1-mteb/RQ2/qwen3-embedding-8b-baseline" \
#     "WinoGrande"


# bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/eval_mteb_single_task_bge.sh \
#     "/data/share/project/shared_models/bge-m3" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-1-mteb/RQ2/bge-m3-baseline" \
#     "SpartQA"

# bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/eval_mteb_single_task_bge.sh \
#     "/data/share/project/shared_models/bge-m3" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-1-mteb/RQ2/bge-m3-baseline" \
#     "WinoGrande"



# # qwen3-8b

# SpartQA_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/spartqa/spartqa-mchoice-20260801-ori-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/spartqa/spartqa-mchoice-20260801-ori-query.jsonl"
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/spartqa/spartqa-mchoice-20260801-ori-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/spartqa/spartqa-mchoice-20260801-ori-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/spartqa/spartqa-mchoice-20260801-gen-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/spartqa/spartqa-mchoice-20260801-gen-query_exclude_other_pos_save_non_pos.jsonl"
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/spartqa/spartqa-mchoice-20260801-gen-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/spartqa/spartqa-mchoice-20260801-gen-query_exclude_other_pos_save_non_pos.jsonl"
# )


# for idx in "${!SpartQA_jsonl_files[@]}"; do
#     jsonl_file="${SpartQA_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="SpartQA"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-1/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/train-qwen3_8b.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/8-1/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-1/RQ2/$task_name/train-qwen3_8b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/eval_mteb_single_task.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/8-1/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-1-mteb/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$task_name"
   
# done




# WinoGrande_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/winogrande/winogrande-20260801-ori-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/winogrande/winogrande-20260801-ori-query_exclude_other_pos_save_non_pos.jsonl"
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/winogrande/winogrande-20260801-ori-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/winogrande/winogrande-20260801-ori-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/winogrande/winogrande-20260801-gen-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/winogrande/winogrande-20260801-gen-query_exclude_other_pos_save_non_pos.jsonl"
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/winogrande/winogrande-20260801-gen-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/origin_data/winogrande/winogrande-20260801-gen-query_exclude_other_pos_save_non_pos.jsonl"
# )

# for idx in "${!WinoGrande_jsonl_files[@]}"; do
#     jsonl_file="${WinoGrande_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="WinoGrande"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-1/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/train-qwen3_8b.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/8-1/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-1/RQ2/$task_name/train-qwen3_8b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/eval_mteb_single_task.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/8-1/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-1-mteb/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$task_name"
   
# done



# # bge-m3

SpartQA_jsonl_files=(
    "/data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/spartqa/spartqa-mchoice-20260801-ori-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/spartqa/spartqa-mchoice-20260801-ori-query.jsonl"
    "/data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/spartqa/spartqa-mchoice-20260801-ori-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/spartqa/spartqa-mchoice-20260801-ori-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/spartqa/spartqa-mchoice-20260801-gen-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/spartqa/spartqa-mchoice-20260801-gen-query_exclude_other_pos_save_non_pos.jsonl"
    "/data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/spartqa/spartqa-mchoice-20260801-gen-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/spartqa/spartqa-mchoice-20260801-gen-query_exclude_other_pos_save_non_pos.jsonl"
)


for idx in "${!SpartQA_jsonl_files[@]}"; do
    jsonl_file="${SpartQA_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="SpartQA"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-1/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/train-bge_m3.sh \
        "/data/share2/project/tr/data_synthesis_agent/model_pth/8-1-debug/RQ2/$task_name/train-bge-v$realidx" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-1-debug/RQ2/$task_name/train-bge-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/eval_mteb_single_task_bge.sh \
        "/data/share2/project/tr/data_synthesis_agent/model_pth/8-1-debug/RQ2/$task_name/train-bge-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/8-1-mteb-debug/RQ2/$task_name/train-bge-v$realidx" \
        "$task_name"
   
done




# WinoGrande_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/winogrande/winogrande-20260801-ori-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/winogrande/winogrande-20260801-ori-query_exclude_other_pos_save_non_pos.jsonl"
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/winogrande/winogrande-20260801-ori-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/winogrande/winogrande-20260801-ori-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/winogrande/winogrande-20260801-gen-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/winogrande/winogrande-20260801-gen-query_exclude_other_pos_save_non_pos.jsonl"
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/winogrande/winogrande-20260801-gen-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/processed_data/winogrande/winogrande-20260801-gen-query_exclude_other_pos_save_non_pos.jsonl"
# )

# for idx in "${!WinoGrande_jsonl_files[@]}"; do
#     jsonl_file="${WinoGrande_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="WinoGrande"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-1/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/train-bge_m3.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/8-1-debug/RQ2/$task_name/train-bge-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-1-debug/RQ2/$task_name/train-bge-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-1/eval_mteb_single_task_bge.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/8-1-debug/RQ2/$task_name/train-bge-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-1-mteb-debug/RQ2/$task_name/train-bge-v$realidx" \
#         "$task_name"
   
# done
