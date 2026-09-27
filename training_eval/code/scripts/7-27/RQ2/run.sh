#!/bin/bash


# bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_bright_singal_task_qwen.sh \
#     "/data/share/project/shared_models/Qwen3-Embedding-8B" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-coir/RQ2/qwen3-embedding-8b-baseline" \
#     "theoremqa_questions"

# bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_bright_singal_task_qwen.sh \
#     "/data/share/project/shared_models/Qwen3-Embedding-8B" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-coir/RQ2/qwen3-embedding-8b-baseline" \
#     "theoremqa_theorems"


# bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_bright_singal_task_bge.sh \
#     "/data/share/project/shared_models/bge-m3" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-coir/RQ2/bge-m3-baseline" \
#     "theoremqa_questions"

# bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_bright_singal_task_bge.sh \
#     "/data/share/project/shared_models/bge-m3" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-coir/RQ2/bge-m3-baseline" \
#     "theoremqa_theorems"



# # qwen3-8b

# theoremqa_questions_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-reason-embed-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-reason-embed-query.jsonl"
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-reason-embed-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-reason-embed-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-gen-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-gen-query.jsonl"
# )


# for idx in "${!theoremqa_questions_jsonl_files[@]}"; do
#     jsonl_file="${theoremqa_questions_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="theoremqa_questions"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name

#     # bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/train-qwen3_8b.sh \
#     #     "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#     #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name/train-qwen3_8b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_bright_singal_task_qwen.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-coir/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$task_name"
   
# done




theoremqa_theorems_jsonl_files=(
    "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260712-reason-embed-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260712-reason-embed-query.jsonl"
    "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260712-reason-embed-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260712-reason-embed-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260713_182321-gen-10k_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260713_182321-gen-10k.jsonl"
)

for idx in "${!theoremqa_theorems_jsonl_files[@]}"; do
    jsonl_file="${theoremqa_theorems_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="theoremqa_theorems"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/train-qwen3_8b.sh \
    #     "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27/RQ2/$task_name/train-qwen3_8b-v$realidx" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name/train-qwen3_8b-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_bright_singal_task_qwen.sh \
         "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27/RQ2/$task_name/train-qwen3_8b-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-coir/RQ2/$task_name/train-qwen3_8b-v$realidx" \
        "$task_name"
   
done



# # bge-m3


# theoremqa_questions_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-reason-embed-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-reason-embed-query.jsonl"
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-reason-embed-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-reason-embed-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-gen-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-gen-query.jsonl"
# )


# for idx in "${!theoremqa_questions_jsonl_files[@]}"; do
#     jsonl_file="${theoremqa_questions_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="theoremqa_questions"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name

#     # bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/train-bge_m3.sh \
#     #     "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27/RQ2/$task_name/train-bge_m3-v$realidx" \
#     #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name/train-bge_m3-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_bright_singal_task_bge.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27/RQ2/$task_name/train-bge_m3-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-coir/RQ2/$task_name/train-bge_m3-v$realidx" \
#         "$task_name"
   
# done




# theoremqa_theorems_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260712-reason-embed-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260712-reason-embed-query.jsonl"
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260712-reason-embed-query_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260712-reason-embed-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260713_182321-gen-10k_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260713_182321-gen-10k.jsonl"
# )

# for idx in "${!theoremqa_theorems_jsonl_files[@]}"; do
#     jsonl_file="${theoremqa_theorems_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="theoremqa_theorems"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name

#     # bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/train-bge_m3.sh \
#     #     "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27/RQ2/$task_name/train-bge_m3-v$realidx" \
#     #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name/train-bge_m3-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_bright_singal_task_bge.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27/RQ2/$task_name/train-bge_m3-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-coir/RQ2/$task_name/train-bge_m3-v$realidx" \
#         "$task_name"
   
# done



# exp-2


# cosqa_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/cosqa/cosqa-0618.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/cosqa/cosqa-0618_exclude_other_pos_save_non_pos.jsonl"
# )



# for idx in "${!cosqa_jsonl_files[@]}"; do
#     jsonl_file="${cosqa_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="cosqa"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/train-bge_m3.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27-test/RQ2/$task_name/train-bge_m3-gen-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-27-test/RQ2/$task_name/train-bge_m3-gen-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_coir_singal_task_bge.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27-test/RQ2/$task_name/train-bge_m3-gen-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-coir-test/RQ2/$task_name/train-bge_m3-gen-v$realidx" \
#         "$task_name"
   
# done




# theoremqa_questions_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-gen-query.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_questions/bright-documents-theoremqa_questions-20260708-gen-query_exclude_other_pos_save_non_pos.jsonl"
# )



# for idx in "${!theoremqa_questions_jsonl_files[@]}"; do
#     jsonl_file="${theoremqa_questions_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="theoremqa_questions"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name

#     # bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/train-bge_m3.sh \
#     #     "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27-test/RQ2/$task_name/train-bge_m3-gen-v$realidx" \
#     #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-27-test/RQ2/$task_name/train-bge_m3-gen-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_bright_singal_task_bge.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27-test/RQ2/$task_name/train-bge_m3-gen-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-bright-test/RQ2/$task_name/train-bge_m3-gen-v$realidx" \
#         "$task_name"
   
# done




# theoremqa_theorems_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260713_182321-gen-10k.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/theoremqa_theorems/bright-documents-theoremqa_theorems-20260713_182321-gen-10k_exclude_other_pos_save_non_pos.jsonl"
# )

# for idx in "${!theoremqa_theorems_jsonl_files[@]}"; do
#     jsonl_file="${theoremqa_theorems_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="theoremqa_theorems"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name

#     # bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/train-bge_m3.sh \
#     #     "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27-test/RQ2/$task_name/train-bge_m3-gen-v$realidx" \
#     #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-27-test/RQ2/$task_name/train-bge_m3-gen-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_bright_singal_task_bge.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27-test/RQ2/$task_name/train-bge_m3-gen-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-bright-test/RQ2/$task_name/train-bge_m3-gen-v$realidx" \
#         "$task_name"
   
# done



# text2sql_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/text2sql/synthetic-text2sql-official-doc-gen-query-0627.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/processed_data/text2sql/synthetic-text2sql-official-doc-gen-query-0627_exclude_other_pos_save_non_pos.jsonl"
# )

# for idx in "${!text2sql_jsonl_files[@]}"; do
#     jsonl_file="${text2sql_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="synthetic-text2sql"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/train-bge_m3.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27-test/RQ2/$task_name/train-bge_m3-gen-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-27-test/RQ2/$task_name/train-bge_m3-gen-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_coir_singal_task_bge.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27-test/RQ2/$task_name/train-bge_m3-gen-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-coir-test/RQ2/$task_name/train-bge_m3-gen-v$realidx" \
#         "$task_name"
   
# done

# others

# cosqa_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/cosqa/cosqa-official-0624.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/cosqa/cosqa-official-0624_exclude_other_pos_save_non_pos.jsonl"
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/cosqa/cosqa-official-0624.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/cosqa/cosqa-official-0624_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/cosqa/cosqa-0618.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/cosqa/cosqa-0618_exclude_other_pos_save_non_pos.jsonl"
# )



# for idx in "${!cosqa_jsonl_files[@]}"; do
#     jsonl_file="${cosqa_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="cosqa"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/train-qwen3_8b.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name/train-qwen3_8b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_coir_singal_task.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-coir/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$task_name"
   
# done




# text2sql_jsonl_files=(
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/text2sql/synthetic-text2sql-official-0627.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/text2sql/synthetic-text2sql-official-0627_exclude_other_pos_save_non_pos.jsonl"
#     "/data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/text2sql/synthetic-text2sql-official-0627.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/text2sql/synthetic-text2sql-official-0627_exclude_other_pos_save_non_pos.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/text2sql/synthetic-text2sql-official-doc-gen-query-0627.jsonl /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/origin_data/text2sql/synthetic-text2sql-official-doc-gen-query-0627_exclude_other_pos_save_non_pos.jsonl"
# )

# for idx in "${!text2sql_jsonl_files[@]}"; do
#     jsonl_file="${text2sql_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="synthetic-text2sql"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/train-qwen3_8b.sh \
#         "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-27/RQ2/$task_name/train-qwen3_8b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-27/RQ2/eval_coir_singal_task.sh \
#          "/data/share2/project/tr/data_synthesis_agent/model_pth/7-27/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/7-27-coir/RQ2/$task_name/train-qwen3_8b-v$realidx" \
#         "$task_name"
   
# done