#!/bin/bash


# bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/eval_mteb_single_task.sh \
#     "/data/share/project/shared_models/Qwen3-Embedding-8B" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-mteb/RQ2/qwen3-embedding-8b-baseline" \
#     "LeCaRDv2"


# bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/eval_mteb_single_task_bge.sh \
#     "/data/share/project/shared_models/bge-m3" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-mteb/RQ2/bge-m3-baseline" \
#     "LeCaRDv2"


# bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/eval_mteb_single_task.sh \
#     "/data/share/project/shared_models/Qwen3-Embedding-8B" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-mteb/RQ2/qwen3-embedding-8b-baseline" \
#     "MedicalRetrieval"


# bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/eval_mteb_single_task_bge.sh \
#     "/data/share/project/shared_models/bge-m3" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-mteb/RQ2/bge-m3-baseline" \
#     "MedicalRetrieval"




# # # qwen3-8b

# LeCaRDv2_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-6/origin_data/lecardv2-20260722-gen-10k.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/origin_data/lecardv2-20260722-gen-10k_exclude_other_pos_save_non_pos.jsonl"
# )


# for idx in "${!LeCaRDv2_jsonl_files[@]}"; do
#     jsonl_file="${LeCaRDv2_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="LeCaRDv2"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/train-qwen3_8b.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/8-6/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name/train-qwen3_8b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/eval_mteb_single_task.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/8-6/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-mteb/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$task_name"
   
# done


# MedicalRetrieval_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-6/origin_data/medical_qa-all-1q-20260806_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/origin_data/medical_qa-all-1q-20260806.jsonl"
# )


# for idx in "${!MedicalRetrieval_jsonl_files[@]}"; do
#     jsonl_file="${MedicalRetrieval_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="MedicalRetrieval"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/train-qwen3_8b.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/8-6/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name/train-qwen3_8b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/eval_mteb_single_task.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/8-6/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-mteb/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$task_name"
   
# done





# # bge-m3

# LeCaRDv2_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-6/processed_data/lecardv2-20260722-gen-10k.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/processed_data/lecardv2-20260722-gen-10k_exclude_other_pos_save_non_pos.jsonl"
# )


# for idx in "${!LeCaRDv2_jsonl_files[@]}"; do
#     jsonl_file="${LeCaRDv2_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="LeCaRDv2"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/train-bge_m3.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/8-6-debug/RQ2/$task_name/train-bge-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-6-debug/RQ2/$task_name/train-bge-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/eval_mteb_single_task_bge.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/8-6-debug/RQ2/$task_name/train-bge-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-mteb-debug/RQ2/$task_name/train-bge-v$realidx" \
#         "$task_name"
   
# done





# MedicalRetrieval_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/8-6/processed_data/medical_qa-all-1q-20260806_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/processed_data/medical_qa-all-1q-20260806.jsonl"
# )


# for idx in "${!MedicalRetrieval_jsonl_files[@]}"; do
#     jsonl_file="${MedicalRetrieval_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="MedicalRetrieval"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/train-bge_m3.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/8-6-debug/RQ2/$task_name/train-bge-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-6-debug/RQ2/$task_name/train-bge-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/eval_mteb_single_task_bge.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/8-6-debug/RQ2/$task_name/train-bge-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-mteb-debug/RQ2/$task_name/train-bge-v$realidx" \
#         "$task_name"
   
# done




bash /data/share/project/tr/science_benchmark/run_flagembedding_eval.sh  \
    "/data/share/project/tr/science_benchmark/converted_full/evidencebench" \
    "/data/share/project/shared_models/Qwen3-Embedding-8B" \
    "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-evidencebench/RQ2/Qwen3-Embedding-8B" \
    --model-class decoder-only-base \
    --query-instruction-format 'Instruct: {}\nQuery: {}' 






evidencebench_jsonl_files=(
    "/data/share/project/tr/data_synthesis_agent/code/scripts/8-6/origin_data/evidencebench-10k-1q-20260806_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/origin_data/evidencebench-10k-1q-20260806.jsonl"
)


for idx in "${!evidencebench_jsonl_files[@]}"; do
    jsonl_file="${evidencebench_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="evidencebench"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-6/train-qwen3_8b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/8-6/RQ2/$task_name/train-qwen3_8b-merged" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-6/RQ2/$task_name/train-qwen3_8b-merged.log

    bash /data/share/project/tr/science_benchmark/run_flagembedding_eval.sh  \
        "/data/share/project/tr/science_benchmark/converted_full/evidencebench" \
        "/data/share/project/tr/data_synthesis_agent/model_pth/8-6/RQ2/$task_name/train-qwen3_8b-merged/merged_model" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/8-6-evidencebench/RQ2/$task_name/train-qwen3_8b-v$realidx-merged" \
        --model-class decoder-only-base \
        --query-instruction-format 'Instruct: {}\nQuery: {}' 
   
done