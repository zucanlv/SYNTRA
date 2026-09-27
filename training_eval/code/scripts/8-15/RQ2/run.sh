#!/bin/bash

# bash /data/share/project/tr/science_benchmark/run_flagembedding_custom_eval.sh  \
#     "legalbench_rag" \
#     "/data/share/project/tr/science_benchmark/converted_flagembedding_custom/legalbench_rag" \
#     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1/merged_model" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/8-15-legalbench_rag-flag-embedding/RQ2/msmarco-baseline/train-qwen3_0.6b-v1"



# codetrans_dl_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/codetrans-dl/codetrans-dl-4q-query1-20260812_exclude_other_pos_save_non_pos.jsonl     /data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/codetrans-dl/codetrans-dl-4q-query1-20260812.jsonl     /data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/codetrans-dl/codetrans-dl-4q-query2-20260812_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/codetrans-dl/codetrans-dl-4q-query2-20260812.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/codetrans-dl/codetrans-dl-4q-query3-20260812_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/codetrans-dl/codetrans-dl-4q-query3-20260812.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/codetrans-dl/codetrans-dl-4q-query4-20260812_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/codetrans-dl/codetrans-dl-4q-query4-20260812.jsonl"
# )


# for idx in "${!codetrans_dl_jsonl_files[@]}"; do
#     jsonl_file="${codetrans_dl_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="codetrans-dl"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-15/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-15/RQ2/train-qwen3_0.6b.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/8-15/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-15/RQ2/$task_name/train-qwen3_0.6b-merged.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-15/RQ2/eval_coir_singal_task.sh  \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/8-15/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx/merged_model" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-15-coir/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged" \
#         "codetrans-dl"
   
# done






# MedicalRetrieval_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medical_qa/medical_qa-all-4q-query1-20260812_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medical_qa/medical_qa-all-4q-query1-20260812.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medical_qa/medical_qa-all-4q-query2-20260812_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medical_qa/medical_qa-all-4q-query2-20260812.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medical_qa/medical_qa-all-4q-query3-20260812_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medical_qa/medical_qa-all-4q-query3-20260812.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medical_qa/medical_qa-all-4q-query4-20260812_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/medical_qa/medical_qa-all-4q-query4-20260812.jsonl"
# )


# for idx in "${!MedicalRetrieval_jsonl_files[@]}"; do
#     jsonl_file="${MedicalRetrieval_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="MedicalRetrieval"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-15/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-15/RQ2/train-qwen3_0.6b.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/8-15/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-15/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-15/RQ2/eval_mteb_single_task.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/8-15/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx/merged_model" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/8-15-mteb/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged" \
#         "$task_name"
   
# done



legalbench_rag_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-1q-20260806_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-1q-20260806.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-2q-query1-20260811_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-2q-query1-20260811.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-2q-query2-20260811_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-2q-query2-20260811.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-4q-query1-20260813_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-4q-query1-20260813.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-4q-query2-20260813_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-4q-query2-20260813.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-4q-query3-20260813_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-4q-query3-20260813.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-4q-query4-20260813_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/legalbench-rag/legalbench-rag-4q-query4-20260813.jsonl"
)


for idx in "${!legalbench_rag_jsonl_files[@]}"; do
    jsonl_file="${legalbench_rag_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="legalbench_rag"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/8-15/RQ2/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/8-15/RQ2/train-qwen3_0.6b.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/8-15/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/8-15/RQ2/$task_name/train-qwen3_0.6b-$realidx-merged.log


    bash /data/share/project/tr/science_benchmark/run_flagembedding_custom_eval.sh  \
        "legalbench_rag" \
        "/data/share/project/tr/science_benchmark/converted_flagembedding_custom/legalbench_rag" \
        "/data/share/project/tr/data_synthesis_agent/model_pth/8-15/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx/merged_model" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/8-15-legalbench_rag-flag-embedding/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged" \



   
done