#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

bash /data/share/project/tr/data_synthesis_agent/code/scripts/7-8/RQ2/eval_bright.sh \
  "/data/share/project/tr/data_synthesis_agent/model_pth/6-20/RQ3_scale/msmarco-10k/train-qwen3_0.6b-v1" \
  "/data/share/project/tr/data_synthesis_agent/eval_output/7-8-bright/RQ3_scale/msmarco-10k-merged/train-qwen3_0.6b-v1"
