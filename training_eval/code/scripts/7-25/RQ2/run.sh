#!/bin/bash

# bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-25/RQ2/eval_mteb_single_task.sh \
#     "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1/merged_model" \
#     "/data/share/project/tr/data_synthesis_agent/eval_output/7-25-mteb/RQ2/msmarco-LeCaRDv2/train-qwen3_0.6b-v1" \
#     "LeCaRDv2"

LeCaRDv2_ori_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/lecardv2/lecardv2-20260722-gen-10k.jsonl /data/share/project/shared_datasets/DSA/syn_data/more-domain-task/lecardv2/lecardv2-20260722-gen-10k_exclude_other_pos_save_non_pos.jsonl"
)


# for idx in "${!LeCaRDv2_ori_jsonl_files[@]}"; do
#     jsonl_file="${LeCaRDv2_ori_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="LeCaRDv2"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-25/RQ2/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-25/RQ2/train-qwen3_0.6b.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/7-25/RQ2/$task_name/train-qwen3_0.6b-merged" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-25/RQ2/$task_name/train-qwen3_0.6b-merged.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-25/RQ2/eval_mteb_single_task.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/7-25/RQ2/$task_name/train-qwen3_0.6b-merged/merged_model" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/7-25-mteb/RQ2/$task_name/train-qwen3_0.6b-v$realidx-merged" \
    #     "$task_name"
   
# done


# msmarco_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/fill-easy-7/msmarco/qwen3-30b/msmarco-gen-10k-0606-fill-easy-7_exclude_other_pos_save_non_pos.jsonl       /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/fill-easy-7/msmarco/qwen3-30b/msmarco-gen-10k-0606-fill-easy-7.jsonl"
#     "      /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/fill-easy-7/msmarco/deepseek-v4-flash/deepseek-v4-flash-20260718_214102-fill-easy-7_exclude_other_pos_save_non_pos.jsonl       /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/fill-easy-7/msmarco/deepseek-v4-flash/deepseek-v4-flash-20260718_214102-fill-easy-7.jsonl"
# )


# for idx in "${!msmarco_jsonl_files[@]}"; do
#     jsonl_file="${msmarco_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="msmarco"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-25/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-25/RQ2/train-qwen3_0.6b.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/7-25/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-25/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-25/RQ2/eval_beir.sh \
#          "/data/share/project/tr/data_synthesis_agent/model_pth/7-25/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/7-25-beir/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$task_name"
   
# done




# text2sql_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/fill-easy-7/synthetic-text2sql/qwen3-30b/qwen3-30b-20260627_080727-fill-easy-7_exclude_other_pos_save_non_pos.jsonl       /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/fill-easy-7/synthetic-text2sql/qwen3-30b/qwen3-30b-20260627_080727-fill-easy-7.jsonl"
#     "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/fill-easy-7/synthetic-text2sql/deepseek-v4-flash/deepseek-v4-flash-20260718_215629-fill-easy-7_exclude_other_pos_save_non_pos.jsonl       /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/fill-easy-7/synthetic-text2sql/deepseek-v4-flash/deepseek-v4-flash-20260718_215629-fill-easy-7.jsonl"
# )

# for idx in "${!text2sql_jsonl_files[@]}"; do
#     jsonl_file="${text2sql_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="synthetic-text2sql"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-25/RQ2/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-25/RQ2/train-qwen3_0.6b.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/7-25/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-25/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-25/RQ2/eval_coir_singal_task.sh \
#          "/data/share/project/tr/data_synthesis_agent/model_pth/7-25/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/7-25-coir/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$task_name"
   
# done

theoremqa_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/fill-easy-7/theoremqa-theorem/qwen3-30b/qwen3-30b-20260713_182321-fill-easy-7_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/fill-easy-7/theoremqa-theorem/qwen3-30b/qwen3-30b-20260713_182321-fill-easy-7.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/fill-easy-7/theoremqa-theorem/deepseek-v4-flash/deepseek-v4-flash-20260720_010413-fill-easy-7_exclude_other_pos_save_non_pos.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ3-scale/scale-size-of-backend-LLMs/fill-easy-7/theoremqa-theorem/deepseek-v4-flash/deepseek-v4-flash-20260720_010413-fill-easy-7.jsonl"
)

for idx in "${!theoremqa_jsonl_files[@]}"; do
    jsonl_file="${theoremqa_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="theoremqa_theorems"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/7-25/RQ2/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-25/RQ2/train-qwen3_0.6b.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/7-25/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/7-25/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx.log

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-25/RQ2/eval_bright_singal_task.sh \
         "/data/share/project/tr/data_synthesis_agent/model_pth/7-25/RQ2/$task_name/train-qwen3_0.6b-merged-v$realidx" \
        "/data/share/project/tr/data_synthesis_agent/eval_output/7-25-bright/RQ2/$task_name/train-qwen3_0.6b-v$realidx" \
        "$task_name"
   
done