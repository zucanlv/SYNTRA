#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

NFCORPUS_ANNOTATED="${PIPELINE_DIR}/Results/nfcorpus_20260615_104328/Annotated_Main_20260615_104405.json"
NFCORPUS_DIVERSE_QUERY="${PIPELINE_DIR}/Results/nfcorpus_20260615_104328/DiverseQuery_FromFile_20260615_104328.json"

OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0615-nfcorpus"
DATA1="${OUTPUT_DIR}/nfcorpus-10k-0615.jsonl"
DATA2="${OUTPUT_DIR}/nfcorpus-10k-0615_exclude_other_pos_save_non_pos.jsonl"

PROMPT="Given a question, retrieve relevant documents that best answer the question."

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing input file: ${path}" >&2
    exit 1
  fi
}

for input_file in \
  "${ANNO_TO_MSMARCO}" \
  "${EXCLUDE_OTHER_POS}" \
  "${NFCORPUS_ANNOTATED}" \
  "${NFCORPUS_DIVERSE_QUERY}"; do
  require_file "${input_file}"
done

mkdir -p "${OUTPUT_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] NF-Corpus: building data1"
python "${ANNO_TO_MSMARCO}" \
  --input "${NFCORPUS_ANNOTATED}" \
  --output "${DATA1}" \
  --prompt "${PROMPT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] NF-Corpus: building data2"
python "${EXCLUDE_OTHER_POS}" \
  --input-jsonl "${DATA1}" \
  --diverse-query "${NFCORPUS_DIVERSE_QUERY}" \
  --output "${DATA2}"

echo
echo "NF-Corpus data1: ${DATA1}"
echo "NF-Corpus data2: ${DATA2}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done."
