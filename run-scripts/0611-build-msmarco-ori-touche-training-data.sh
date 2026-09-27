#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"
COUNT_POS_NEG="${PREPROCESS_DIR}/4-21-count-training-data-pos-neg.py"

MSMARCO_ANNOTATED="${PIPELINE_DIR}/Results/msmarco_20260610_180656/Annotated_Main_20260610_180927.json"
MSMARCO_DIVERSE_QUERY="${PIPELINE_DIR}/Results/msmarco_20260610_180656/DiverseQuery_FromFile_20260610_180657.json"
TOUCHE_ANNOTATED="${PIPELINE_DIR}/Results/touche2020_20260610_223346/Annotated_Main_20260610_223402.json"
TOUCHE_DIVERSE_QUERY="${PIPELINE_DIR}/Results/touche2020_20260606_220417/DiverseQuery_Main_20260606_233127.json"

MSMARCO_OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori/0606/ori"
TOUCHE_OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0611-touche"

MSMARCO_DATA1="${MSMARCO_OUTPUT_DIR}/msmarco-ori-10k-0611.jsonl"
MSMARCO_DATA2="${MSMARCO_OUTPUT_DIR}/msmarco-ori-10k-0611_exclude_other_pos_save_non_pos.jsonl"
TOUCHE_DATA1="${TOUCHE_OUTPUT_DIR}/touche-10k-0611.jsonl"
TOUCHE_DATA2="${TOUCHE_OUTPUT_DIR}/touche-10k-0611_exclude_other_pos_save_non_pos.jsonl"
TOUCHE_POS_NEG_PNG="${TOUCHE_OUTPUT_DIR}/touche-10k-0611_exclude_other_pos_save_non_pos_posneg.png"

MSMARCO_PROMPT="Given a web search query, retrieve relevant passages that answer the query."
TOUCHE_PROMPT="Given a question, retrieve detailed and persuasive arguments that answer the question."

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing input file: ${path}" >&2
    exit 1
  fi
}

for input_file in \
  "${MSMARCO_ANNOTATED}" \
  "${MSMARCO_DIVERSE_QUERY}" \
  "${TOUCHE_ANNOTATED}" \
  "${TOUCHE_DIVERSE_QUERY}"; do
  require_file "${input_file}"
done

mkdir -p "${MSMARCO_OUTPUT_DIR}" "${TOUCHE_OUTPUT_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] MS MARCO ori: building data1"
python "${ANNO_TO_MSMARCO}" \
  --input "${MSMARCO_ANNOTATED}" \
  --output "${MSMARCO_DATA1}" \
  --prompt "${MSMARCO_PROMPT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] MS MARCO ori: building data2"
python "${EXCLUDE_OTHER_POS}" \
  --input-jsonl "${MSMARCO_DATA1}" \
  --diverse-query "${MSMARCO_DIVERSE_QUERY}" \
  --output "${MSMARCO_DATA2}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Touché: building data1"
python "${ANNO_TO_MSMARCO}" \
  --input "${TOUCHE_ANNOTATED}" \
  --output "${TOUCHE_DATA1}" \
  --prompt "${TOUCHE_PROMPT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Touché: building data2"
python "${EXCLUDE_OTHER_POS}" \
  --input-jsonl "${TOUCHE_DATA1}" \
  --diverse-query "${TOUCHE_DIVERSE_QUERY}" \
  --output "${TOUCHE_DATA2}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Touché: plotting pos/neg distribution"
python "${COUNT_POS_NEG}" \
  "${TOUCHE_DATA2}" \
  --bin-width 1 \
  --save-png "${TOUCHE_POS_NEG_PNG}"

echo
echo "MS MARCO data1 : ${MSMARCO_DATA1}"
echo "MS MARCO data2 : ${MSMARCO_DATA2}"
echo "Touché data1   : ${TOUCHE_DATA1}"
echo "Touché data2   : ${TOUCHE_DATA2}"
echo "Touché pos/neg : ${TOUCHE_POS_NEG_PNG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done."
