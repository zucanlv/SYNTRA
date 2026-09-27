#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/Main_Pipeline"
PREPROCESS_DIR="${PIPELINE_DIR}/corpus_preprocess"

CODE_OUTPUT_DIR="${CODE_OUTPUT_DIR:-/data/share/project/shared_datasets/DSA/syn_data/RQ2-code-retrieve-task/codetrans-dl}"
DOMAIN_OUTPUT_ROOT="${DOMAIN_OUTPUT_ROOT:-/data/share/project/shared_datasets/DSA/syn_data/more-domain-task}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

ANNO_TO_MSMARCO="${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py"
EXCLUDE_OTHER_POS="${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py"

CODETRANS_PROMPT="Given a TensorFlow code snippet, retrieve functionally equivalent PaddlePaddle code snippets that implement the same deep learning functionality."
CQADUPSTACK_MATHEMATICA_PROMPT="Given a Mathematica Q&A question, retrieve relevant Mathematica Q&A forum posts that ask or address the same underlying question or problem."
LEGALBENCH_RAG_PROMPT="Given a question about a legal contract or privacy policy, retrieve relevant contract or policy passages that directly answer it or provide decisive evidence."
EVIDENCEBENCH_PROMPT="Given a biomedical scientific hypothesis, retrieve relevant sentences from biomedical research papers that support, contradict, or qualify the hypothesis."
MEDICAL_QA_PROMPT="Given an English-language medical question, retrieve relevant English-language medical answer passages that directly answer the question."
NQ_PROMPT="Given a natural-language factoid question, retrieve relevant Wikipedia passages that contain the factual evidence needed to answer the question."

NAMES=(
  "codetrans-dl-1q-20260806"
  "codetrans-dl-2q-20260806"
  "cqadupstack-mathematica-10k-20260805"
  "legalbench-rag-1q-20260806"
  "legalbench-rag-2q-20260806"
  "evidencebench-10k-1q-20260806"
  "medical_qa-all-1q-20260806"
  "medical_qa-all-2q-20260806"
  "nq-10k-1q-20260806"
)

ANNOTATED_FILES=(
  "${PIPELINE_DIR}/Results/codetrans-dl_20260806_013902/Annotated_Main_20260806_021433.json"
  "${PIPELINE_DIR}/Results/codetrans-dl_20260806_013936/Annotated_Main_20260806_022126.json"
  "${PIPELINE_DIR}/Results/cqadupstack-mathematica_20260805_172240/Annotated_Main_20260805_172252.json"
  "${PIPELINE_DIR}/Results/legalbench-rag_20260806_014143/Annotated_Main_20260806_030301.json"
  "${PIPELINE_DIR}/Results/legalbench-rag_20260806_014204/Annotated_Main_20260806_030912.json"
  "${PIPELINE_DIR}/Results/evidencebench_20260806_065436/Annotated_Main_20260806_065439.json"
  "${PIPELINE_DIR}/Results/medical_qa_20260806_070242/Annotated_Main_20260806_090415.json"
  "${PIPELINE_DIR}/Results/medical_qa_20260806_070404/Annotated_Main_20260806_090558.json"
  "${PIPELINE_DIR}/Results/nq_20260806_013724/Annotated_Main_20260806_092944.json"
)

DIVERSE_QUERY_FILES=(
  "${PIPELINE_DIR}/Results/codetrans-dl_20260806_013902/DiverseQuery_Main_20260806_021311.json"
  "${PIPELINE_DIR}/Results/codetrans-dl_20260806_013936/DiverseQuery_Main_20260806_021937.json"
  "${PIPELINE_DIR}/Results/cqadupstack-mathematica_20260805_151125/DiverseQuery_FromFile_20260805_151126.json"
  "${PIPELINE_DIR}/Results/legalbench-rag_20260806_014143/DiverseQuery_Main_20260806_025715.json"
  "${PIPELINE_DIR}/Results/legalbench-rag_20260806_014204/DiverseQuery_Main_20260806_025811.json"
  "${PIPELINE_DIR}/Results/evidencebench_20260806_040833/DiverseQuery_Main_20260806_053348.json"
  "${PIPELINE_DIR}/Results/medical_qa_20260806_070242/DiverseQuery_Main_20260806_090338.json"
  "${PIPELINE_DIR}/Results/medical_qa_20260806_070404/DiverseQuery_Main_20260806_090515.json"
  "${PIPELINE_DIR}/Results/nq_20260806_013724/DiverseQuery_Main_20260806_092738.json"
)

OUTPUT_DIRS=(
  "${CODE_OUTPUT_DIR}"
  "${CODE_OUTPUT_DIR}"
  "${DOMAIN_OUTPUT_ROOT}/cqadupstack-mathematica"
  "${DOMAIN_OUTPUT_ROOT}/legalbench-rag"
  "${DOMAIN_OUTPUT_ROOT}/legalbench-rag"
  "${DOMAIN_OUTPUT_ROOT}/evidencebench"
  "${DOMAIN_OUTPUT_ROOT}/medical_qa"
  "${DOMAIN_OUTPUT_ROOT}/medical_qa"
  "${DOMAIN_OUTPUT_ROOT}/nq"
)

PROMPTS=(
  "${CODETRANS_PROMPT}"
  "${CODETRANS_PROMPT}"
  "${CQADUPSTACK_MATHEMATICA_PROMPT}"
  "${LEGALBENCH_RAG_PROMPT}"
  "${LEGALBENCH_RAG_PROMPT}"
  "${EVIDENCEBENCH_PROMPT}"
  "${MEDICAL_QA_PROMPT}"
  "${MEDICAL_QA_PROMPT}"
  "${NQ_PROMPT}"
)

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

run_command() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

array_length="${#NAMES[@]}"
if [[ "${array_length}" -ne "${#ANNOTATED_FILES[@]}" ]] \
  || [[ "${array_length}" -ne "${#DIVERSE_QUERY_FILES[@]}" ]] \
  || [[ "${array_length}" -ne "${#OUTPUT_DIRS[@]}" ]] \
  || [[ "${array_length}" -ne "${#PROMPTS[@]}" ]]; then
  echo "Internal error: build arrays have different lengths." >&2
  exit 2
fi

require_file "${ANNO_TO_MSMARCO}"
require_file "${EXCLUDE_OTHER_POS}"
for i in "${!NAMES[@]}"; do
  require_file "${ANNOTATED_FILES[$i]}"
  require_file "${DIVERSE_QUERY_FILES[$i]}"
done

if [[ "${DRY_RUN}" != "1" && "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/ustc/public_envs/DSA
fi

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  annotated_file="${ANNOTATED_FILES[$i]}"
  diverse_query_file="${DIVERSE_QUERY_FILES[$i]}"
  output_dir="${OUTPUT_DIRS[$i]}"
  prompt="${PROMPTS[$i]}"
  training_data="${output_dir}/${name}.jsonl"
  filtered_training_data="${output_dir}/${name}_exclude_other_pos_save_non_pos.jsonl"

  run_command mkdir -p "${output_dir}"

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building ${name}"
  echo "  annotated    : ${annotated_file}"
  echo "  diverse query: ${diverse_query_file}"
  echo "  prompt       : ${prompt}"
  echo "  output       : ${training_data}"
  echo "  filtered     : ${filtered_training_data}"

  run_command \
    python "${ANNO_TO_MSMARCO}" \
    --input "${annotated_file}" \
    --output "${training_data}" \
    --prompt "${prompt}"

  run_command \
    python "${EXCLUDE_OTHER_POS}" \
    --input-jsonl "${training_data}" \
    --diverse-query "${diverse_query_file}" \
    --output "${filtered_training_data}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN complete: validated ${array_length} ordinary training-data builds"
else
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All ${array_length} ordinary training-data builds are ready."
fi
