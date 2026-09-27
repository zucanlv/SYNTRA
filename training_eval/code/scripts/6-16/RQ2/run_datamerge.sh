#!/bin/bash
# scidocs_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0614-scidocs/scidocs-10k-0614-fill-easy.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0614-scidocs/scidocs-10k-0614-fill-easy_exclude_other_pos_save_non_pos.jsonl"
# )

# for idx in "${!scidocs_jsonl_files[@]}"; do
#     jsonl_file="${scidocs_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="scidocs"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-16/RQ2_general_task/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/train-qwen3_0.6b_datamerge.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/eval_beir_single_task.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/6-16-beir/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$task_name"

# done


# nfcorpus_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0615-nfcorpus/nfcorpus-10k-0615.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0615-nfcorpus/nfcorpus-10k-0615_exclude_other_pos_save_non_pos.jsonl"
# )

# for idx in "${!nfcorpus_jsonl_files[@]}"; do
#     jsonl_file="${nfcorpus_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="nfcorpus"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-16/RQ2_general_task/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/train-qwen3_0.6b_datamerge.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/eval_beir_single_task.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/6-16-beir/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$task_name"

# done

# figa_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0614-fiqa/fiqa-10k-0614.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0614-fiqa/fiqa-10k-0614_exclude_other_pos_save_non_pos.jsonl"
# )

# for idx in "${!figa_jsonl_files[@]}"; do
#     jsonl_file="${figa_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="figa"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-17/RQ2_general_task/$task_name

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/train-qwen3_0.6b_datamerge.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-17/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-17/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/eval_beir_single_task.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/6-17/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/6-17-beir/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$task_name"

# done

# dbpedia_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0604/dbpedia-10k-0604-fewshot-cali-fill-easy-7.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0604/dbpedia-10k-0604-fewshot-cali-fill-easy-7_exclude_others_save_non_pos.jsonl"
# )

# for idx in "${!dbpedia_jsonl_files[@]}"; do
#     jsonl_file="${dbpedia_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="dbpedia-entity"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-16/RQ2_general_task/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/train-qwen3_0.6b_datamerge.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/eval_beir_single_task.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/6-16-beir/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$task_name"

# done



# touche2020_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0611-touche/touche-10k-0611.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0611-touche/touche-10k-0611_exclude_other_pos_save_non_pos.jsonl"
# )

# for idx in "${!touche2020_jsonl_files[@]}"; do
#     jsonl_file="${touche2020_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="webis-touche2020"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-16/RQ2_general_task/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/train-qwen3_0.6b_datamerge.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/eval_beir_single_task.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/6-16-beir/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$task_name"

# done

# nfcorpus_jsonl_files=(
#     "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0615-nfcorpus/nfcorpus-10k-0615.jsonl"
#     "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0615-nfcorpus/nfcorpus-10k-0615_exclude_other_pos_save_non_pos.jsonl"
# )

# for idx in "${!nfcorpus_jsonl_files[@]}"; do
#     jsonl_file="${nfcorpus_jsonl_files[$idx]}"
#     echo "running idx=$idx jsonl_file=$jsonl_file"
#     realidx=$((idx + 1))
#     task_name="nfcorpus"
#     mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-17/RQ2_general_task/$task_name

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/train-qwen3_0.6b.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/6-17/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-17/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx.log

#     bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/eval_beir_single_task.sh \
#         "/data/share/project/tr/data_synthesis_agent/model_pth/6-17/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "/data/share/project/tr/data_synthesis_agent/eval_output/6-17-beir/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
#         "$task_name"

# done

msmarco_jsonl_files=(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori/0606/gen/msmarco-gen-10k-0606.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori/0606/gen/msmarco-gen-10k-0606_exclude_other_pos_save_non_pos.jsonl"
    "/data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori/0606/ori/msmarco-ori-10k-0611.jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori/0606/ori/msmarco-ori-10k-0611_exclude_other_pos_save_non_pos.jsonl"
)

for idx in "${!msmarco_jsonl_files[@]}"; do
    jsonl_file="${msmarco_jsonl_files[$idx]}"
    echo "running idx=$idx jsonl_file=$jsonl_file"
    realidx=$((idx + 1))
    task_name="msmarco"
    mkdir -p /data/share/project/tr/data_synthesis_agent/logs/6-16/RQ2_general_task/$task_name

    bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/train-qwen3_0.6b_datamerge.sh \
        "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
        "$jsonl_file" 2>&1 | tee /data/share/project/tr/data_synthesis_agent/logs/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx.log

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/eval_beir_qwen3_0.6b-msmarco.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-16-msmarco/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx"

    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/eval_nano_beir.sh \
    #     "/data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-16-nanobeir/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx"
    
    # python /data/share/project/tr/data_synthesis_agent/code/scripts/6-16/RQ2/summary.py \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/6-16-nanobeir/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx"/*/* \
    #     > "/data/share/project/tr/data_synthesis_agent/eval_output/6-16-nanobeir/RQ2_general_task/$task_name/train-qwen3_0.6b-v$realidx/summary.md"
    
done
