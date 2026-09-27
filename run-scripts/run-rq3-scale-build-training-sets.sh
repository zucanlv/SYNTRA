#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"
OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ3-scale"
PROMPT="Given a web search query, retrieve relevant passages that answer the query."

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

mkdir -p "${OUTPUT_DIR}"

NAMES=(
  "msmarco-gen-0k-40k"
  "msmarco-gen-40k-60k"
  "msmarco-gen-60k-120k"
  "msmarco-gen-120k-150k"
  "msmarco-gen-150k-180k"
  "msmarco-gen-180k-210k"
  "msmarco-gen-210k-240k"
  "msmarco-gen-240k-270k"
  "msmarco-gen-270k-300k"
  "msmarco-gen-300k-330k"
  "msmarco-gen-330k-360k"
)

ANNOTATED_FILES=(
  "${PIPELINE_DIR}/Results/msmarco_20260508_075045/Annotated_Main_20260508_162108.json"
  "${PIPELINE_DIR}/Results/msmarco_20260508_075025/Annotated_Main_20260508_120802.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_063619/Annotated_Main_20260509_175525.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_182643/Annotated_Main_20260510_012341.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_182735/Annotated_Main_20260510_013347.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_152634/Annotated_Main_20260510_152648.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_182816/Annotated_Main_20260510_015112.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_152659/Annotated_Main_20260510_152713.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_152718/Annotated_Main_20260510_152735.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_152741/Annotated_Main_20260510_152755.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_152804/Annotated_Main_20260510_152818.json"
)

DIVERSE_QUERY_FILES=(
  "${PIPELINE_DIR}/Results/msmarco_20260508_075045/DiverseQuery_Main_20260508_160631.json"
  "${PIPELINE_DIR}/Results/msmarco_20260508_075025/DiverseQuery_Main_20260508_120033.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_063619/DiverseQuery_Main_20260509_173727.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_182643/DiverseQuery_Main_20260510_011420.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_182735/DiverseQuery_Main_20260510_011458.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_143942/DiverseQuery_FromFile_20260510_143944.json"
  "${PIPELINE_DIR}/Results/msmarco_20260509_182816/DiverseQuery_Main_20260510_011537.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_151052/DiverseQuery_FromFile_20260510_151054.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_151202/DiverseQuery_FromFile_20260510_151204.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_144239/DiverseQuery_FromFile_20260510_144241.json"
  "${PIPELINE_DIR}/Results/msmarco_20260510_144346/DiverseQuery_FromFile_20260510_144348.json"
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All RQ3-scale training sets are ready."
