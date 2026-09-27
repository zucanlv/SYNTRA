#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"

ANNO_FILL_EASY="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn_fill_easy.py"
ANNO_STANDARD="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

SCIDOCS_ANNOTATED="${PIPELINE_DIR}/Results/scidocs_20260614_084708/Annotated_Main_20260614_102735.json"
SCIDOCS_DIVERSE_QUERY="${PIPELINE_DIR}/Results/scidocs_20260614_084708/DiverseQuery_Main_20260614_102605.json"
FIQA_ANNOTATED="${PIPELINE_DIR}/Results/fiqa_20260613_180752/Annotated_Main_20260613_194206.json"
FIQA_DIVERSE_QUERY="${PIPELINE_DIR}/Results/fiqa_20260613_180752/DiverseQuery_Main_20260613_194057.json"

SCIDOCS_OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0614-scidocs"
FIQA_OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0614-fiqa"

SCIDOCS_DATA1="${SCIDOCS_OUTPUT_DIR}/scidocs-10k-0614-fill-easy.jsonl"
SCIDOCS_DATA2="${SCIDOCS_OUTPUT_DIR}/scidocs-10k-0614-fill-easy_exclude_other_pos_save_non_pos.jsonl"
FIQA_DATA1="${FIQA_OUTPUT_DIR}/fiqa-10k-0614.jsonl"
FIQA_DATA2="${FIQA_OUTPUT_DIR}/fiqa-10k-0614_exclude_other_pos_save_non_pos.jsonl"

SCIDOCS_PROMPT="Given a scientific paper title, retrieve paper abstracts that are cited by the given paper."
FIQA_PROMPT="Given a financial question, retrieve user replies that best answer the question."
SCIDOCS_TARGET_NEGATIVES=7

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing input file: ${path}" >&2
    exit 1
  fi
}

for input_file in \
  "${ANNO_FILL_EASY}" \
  "${ANNO_STANDARD}" \
  "${EXCLUDE_OTHER_POS}" \
  "${SCIDOCS_ANNOTATED}" \
  "${SCIDOCS_DIVERSE_QUERY}" \
  "${FIQA_ANNOTATED}" \
  "${FIQA_DIVERSE_QUERY}"; do
  require_file "${input_file}"
done

mkdir -p "${SCIDOCS_OUTPUT_DIR}" "${FIQA_OUTPUT_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] SciDocs: building data1 with easy-negative fill"
python "${ANNO_FILL_EASY}" \
  --input "${SCIDOCS_ANNOTATED}" \
  --output "${SCIDOCS_DATA1}" \
  --prompt "${SCIDOCS_PROMPT}" \
  --target-negatives "${SCIDOCS_TARGET_NEGATIVES}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] SciDocs: building data2"
python "${EXCLUDE_OTHER_POS}" \
  --input-jsonl "${SCIDOCS_DATA1}" \
  --diverse-query "${SCIDOCS_DIVERSE_QUERY}" \
  --output "${SCIDOCS_DATA2}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] FiQA: building data1"
python "${ANNO_STANDARD}" \
  --input "${FIQA_ANNOTATED}" \
  --output "${FIQA_DATA1}" \
  --prompt "${FIQA_PROMPT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] FiQA: building data2"
python "${EXCLUDE_OTHER_POS}" \
  --input-jsonl "${FIQA_DATA1}" \
  --diverse-query "${FIQA_DIVERSE_QUERY}" \
  --output "${FIQA_DATA2}"

echo
echo "SciDocs data1: ${SCIDOCS_DATA1}"
echo "SciDocs data2: ${SCIDOCS_DATA2}"
echo "FiQA data1   : ${FIQA_DATA1}"
echo "FiQA data2   : ${FIQA_DATA2}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done."
