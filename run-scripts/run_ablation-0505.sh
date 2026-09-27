#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/ustc/public_envs/DSA

cd /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ablation-2-query-gen-instr"
python main.py --config config/config_0505-msmarco-ablation-2-query-gen-instr.yaml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ablation-3-idattr"
python main.py --config config/config_0505-msmarco-ablation-3-idattr.yaml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ablation-4-hard-neg-mining"
python main.py --config config/config_0505-msmarco-ablation-4-hard-neg-mining.yaml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ablation-5-without-self-refine"
python main.py --config config/config_0505-msmarco-ablation-5-without-self-refine.yaml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done."
