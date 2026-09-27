#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-19/eval_beir.sh \
    /data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/msmarco/train-qwen3_0.6b-v1/merged_model \
    /data/share/project/tr/data_synthesis_agent/eval_output/6-19-beir/train-qwen3_0.6b-v1 \

bash /data/share/project/tr/data_synthesis_agent/code/scripts/6-19/eval_beir.sh \
    /data/share/project/tr/data_synthesis_agent/model_pth/6-16/RQ2_general_task/msmarco/train-qwen3_0.6b-v2/merged_model \
    /data/share/project/tr/data_synthesis_agent/eval_output/6-19-beir/train-qwen3_0.6b-v2 \
