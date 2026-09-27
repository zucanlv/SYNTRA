#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

cd /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: msmarco-10k-gen"
NO_PROXY=127.0.0.1,localhost,0.0.0.0 no_proxy=127.0.0.1,localhost,0.0.0.0 \
  python main.py --config config/config-0606-msmarco-10k-gen.yaml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: msmarco-10k-ori"
NO_PROXY=127.0.0.1,localhost,0.0.0.0 no_proxy=127.0.0.1,localhost,0.0.0.0 \
  python main.py --config config/config-0606-msmarco-10k-ori.yaml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: touche-10k"
NO_PROXY=127.0.0.1,localhost,0.0.0.0 no_proxy=127.0.0.1,localhost,0.0.0.0 \
  python main.py --config config/config-0606-touche-10k.yaml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done."
