#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"
OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori"
PROMPT="Given a web search query, retrieve relevant passages that answer the query."

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

mkdir -p "${OUTPUT_DIR}"

NAMES=(
  "msmarco-gen-0k-24k"
  "msmarco-gen-24k-48k"
  "msmarco-gen-48k-74k"
)

ANNOTATED_FILES=(
  "${PIPELINE_DIR}/Results/msmarco_20260510_055221/Annotated_Main_20260510_055234.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_055421/Annotated_Main_20260510_055433.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_055621/Annotated_Main_20260510_055634.json"
)

DIVERSE_QUERY_FILES=(
  "${PIPELINE_DIR}/Results/msmarco_20260509_173533/DiverseQuery_Main_20260510_010116.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_173732/DiverseQuery_Main_20260510_013136.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_173932/DiverseQuery_Main_20260510_014007.json"
)

echo "Output directory: ${OUTPUT_DIR}"

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  annotated_file="${ANNOTATED_FILES[$i]}"
  diverse_query_file="${DIVERSE_QUERY_FILES[$i]}"
  training_set="${OUTPUT_DIR}/${name}.jsonl"
  training_set_filtered="${OUTPUT_DIR}/${name}_exclude_other_pos_save_non_pos.jsonl"

  if [[ ! -f "${annotated_file}" ]]; then
    echo "Missing Annotated file: ${annotated_file}" >&2
    exit 1
  fi
  if [[ ! -f "${diverse_query_file}" ]]; then
    echo "Missing DiverseQuery file: ${diverse_query_file}" >&2
    exit 1
  fi

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1: ${name}"
  python "${ANNO_TO_MSMARCO}" \
    --input "${annotated_file}" \
    --output "${training_set}" \
    --prompt "${PROMPT}"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2: ${name}"
  python "${EXCLUDE_OTHER_POS}" \
    --input-jsonl "${training_set}" \
    --diverse-query "${diverse_query_file}" \
    --output "${training_set_filtered}"

  echo "First training set : ${training_set}"
  echo "Second training set: ${training_set_filtered}"
done

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All RQ1-gen-ori training sets are ready."
