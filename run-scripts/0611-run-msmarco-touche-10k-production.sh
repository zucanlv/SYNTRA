#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
cd "${PROJECT_ROOT}/Main_Pipeline"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: msmarco-10k-ori"
NO_PROXY=127.0.0.1,localhost,0.0.0.0 no_proxy=127.0.0.1,localhost,0.0.0.0 \
  python main.py --config config/config-0611-msmarco-10k-ori.yaml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: touche-10k-production"
NO_PROXY=127.0.0.1,localhost,0.0.0.0 no_proxy=127.0.0.1,localhost,0.0.0.0 \
  python main.py --config config/config-0611-touche-10k-production.yaml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done."
