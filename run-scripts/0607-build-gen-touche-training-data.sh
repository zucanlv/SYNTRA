#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

# If the run directories differ, edit only these two variables.
MSMARCO_GEN_RESULT_DIR="${PIPELINE_DIR}/Results/msmarco_20260606_161638"
TOUCHE_RESULT_DIR="${PIPELINE_DIR}/Results/touche2020_20260606_220417"

MSMARCO_GEN_OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori/0606/gen"
TOUCHE_OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0606-touche"

MSMARCO_PROMPT="Given a web search query, retrieve relevant passages that answer the query."
TOUCHE_PROMPT="Given a question, retrieve detailed and persuasive arguments that answer the question."

latest_one() {
  local dir="$1"
  local pattern="$2"
  find "${dir}" -maxdepth 1 -type f -name "${pattern}" -printf '%T@ %p\n' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ -z "${path}" || ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

build_dataset() {
  local name="$1"
  local result_dir="$2"
  local output_dir="$3"
  local output_base="$4"
  local prompt="$5"

  local annotated_file
  local diverse_query_file
  local data1
  local data2

  annotated_file="$(latest_one "${result_dir}" 'Annotated_*.json')"
  diverse_query_file="$(latest_one "${result_dir}" 'DiverseQuery_*.json')"
  data1="${output_dir}/${output_base}.jsonl"
  data2="${output_dir}/${output_base}_exclude_other_pos_save_non_pos.jsonl"

  require_file "${annotated_file}" "${name} annotated file"
  require_file "${diverse_query_file}" "${name} diverse-query file"
  mkdir -p "${output_dir}"

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name}: data1"
  echo "  Annotated    : ${annotated_file}"
  echo "  Output data1 : ${data1}"
  python "${ANNO_TO_MSMARCO}" \
    --input "${annotated_file}" \
    --output "${data1}" \
    --prompt "${prompt}"

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name}: data2"
  echo "  DiverseQuery : ${diverse_query_file}"
  echo "  Output data2 : ${data2}"
  python "${EXCLUDE_OTHER_POS}" \
    --input-jsonl "${data1}" \
    --diverse-query "${diverse_query_file}" \
    --output "${data2}"
}

build_dataset \
  "msmarco-gen-0606" \
  "${MSMARCO_GEN_RESULT_DIR}" \
  "${MSMARCO_GEN_OUTPUT_DIR}" \
  "msmarco-gen-10k-0606" \
  "${MSMARCO_PROMPT}"

build_dataset \
  "touche-0606" \
  "${TOUCHE_RESULT_DIR}" \
  "${TOUCHE_OUTPUT_DIR}" \
  "touche-10k-0606" \
  "${TOUCHE_PROMPT}"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done."
