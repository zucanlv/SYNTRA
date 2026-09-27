#!/bin/bash
gen_model_path=(
    "/data/share/project/tr/data_synthesis_agent/model_pth/4-27/gen/train-qwen3_0.6b-msmarco_anno-v1/merged_model"
    "/data/share/project/tr/data_synthesis_agent/model_pth/4-30/gen_exclude_other_pos_save_non_pos/train-qwen3_0.6b-msmarco_anno-v1/merged_model"
)
ori_model_path=(
    "/data/share/project/tr/data_synthesis_agent/model_pth/4-18/ori/train-qwen3_0.6b-msmarco_anno-v1/merged_model"
    "/data/share/project/tr/data_synthesis_agent/model_pth/5-3/gen_exclude_other_pos/train-qwen3_0.6b-msmarco_anno-v1/merged_model"
)

for idx in "${!ori_model_path[@]}"; do
    echo "running idx=$idx"
    realidx=$((idx + 1))
    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/nano_beir_gen_ori/eval_nano_beir.sh \
    #     "${ori_model_path[$idx]}" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/5-6/nano_beir-msmarco-ori-test/train-msmarco-anno_v$realidx"
    python   /data/share/project/tr/mmteb/code/train/qwen3-8b/code/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6/nano_beir-msmarco-ori-test/train-msmarco-anno_v$realidx/no_model_name_available/no_revision_available" \
         > "/data/share/project/tr/data_synthesis_agent/eval_output/5-6/nano_beir-msmarco-ori-test/train-msmarco-anno_v$realidx/summary.md"
done

for idx in "${!gen_model_path[@]}"; do
    echo "running idx=$idx"
    realidx=$((idx + 1))
    # bash /data/share/project/tr/data_synthesis_agent/code/scripts/5-6/nano_beir_gen_ori/eval_nano_beir.sh \
    #     "${gen_model_path[$idx]}" \
    #     "/data/share/project/tr/data_synthesis_agent/eval_output/5-6/nano_beir-msmarco-gen-test/train-msmarco-anno_v$realidx"
    python   /data/share/project/tr/mmteb/code/train/qwen3-8b/code/summary.py \
        "/data/share/project/tr/data_synthesis_agent/eval_output/5-6/nano_beir-msmarco-gen-test/train-msmarco-anno_v$realidx/no_model_name_available/no_revision_available" \
         > "/data/share/project/tr/data_synthesis_agent/eval_output/5-6/nano_beir-msmarco-gen-test/train-msmarco-anno_v$realidx/summary.md"

done