#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/ustc/public_envs/DSA

cd /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: dt-sample"
python main.py --config config/config_0502-msmarco-ablation-dt-selection-dt-sample.yaml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: random-sample"
python main.py --config config/config_0502-msmarco-ablation-dt-selection-random-sample.yaml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done."
