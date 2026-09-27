#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
PREPROCESS_DIR="${PROJECT_ROOT}/Main_Pipeline/corpus_preprocess"
SPLIT_SCRIPT="${PREPROCESS_DIR}/6-16-split-training-pos-by-median.py"

MSMARCO_GEN_JSONL="/data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori/0606/gen/msmarco-gen-10k-0606.jsonl"
MSMARCO_GEN_DIVERSE_QUERY="${PROJECT_ROOT}/Main_Pipeline/Results/msmarco_20260606_161638/DiverseQuery_Main_20260606_173645.json"

MSMARCO_ORI_JSONL="/data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori/0606/ori/msmarco-ori-10k-0611.jsonl"
MSMARCO_ORI_DIVERSE_QUERY="${PROJECT_ROOT}/Main_Pipeline/Results/msmarco_20260610_180656/DiverseQuery_FromFile_20260610_180657.json"

DBPEDIA_JSONL="/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0604/dbpedia-10k-0604-fewshot-cali-fill-easy-7.jsonl"
DBPEDIA_DIVERSE_QUERY="${PROJECT_ROOT}/Main_Pipeline/Results/dbpedia_20260604_025215/DiverseQuery_FromFile_20260604_025216.json"

TOUCHE2020_JSONL="/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0611-touche/touche-10k-0611.jsonl"
TOUCHE2020_DIVERSE_QUERY="${PROJECT_ROOT}/Main_Pipeline/Results/touche2020_20260606_220417/DiverseQuery_Main_20260606_233127.json"

FIQA_JSONL="/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0614-fiqa/fiqa-10k-0614.jsonl"
FIQA_DIVERSE_QUERY="${PROJECT_ROOT}/Main_Pipeline/Results/fiqa_20260613_180752/DiverseQuery_Main_20260613_194057.json"

SCIDOCS_JSONL="/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0614-scidocs/scidocs-10k-0614-fill-easy.jsonl"
SCIDOCS_DIVERSE_QUERY="${PROJECT_ROOT}/Main_Pipeline/Results/scidocs_20260614_084708/DiverseQuery_Main_20260614_102605.json"

NFCORPUS_JSONL="/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0615-nfcorpus/nfcorpus-10k-0615.jsonl"
NFCORPUS_DIVERSE_QUERY="${PROJECT_ROOT}/Main_Pipeline/Results/nfcorpus_20260615_104328/DiverseQuery_FromFile_20260615_104328.json"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing input file: ${path}" >&2
    exit 1
  fi
}

run_split() {
  local name="$1"
  local input_jsonl="$2"
  local diverse_query="$3"
  local output_dir="${input_jsonl%.jsonl}_median_pos_splits"
  shift 3
  local extra_args=("$@")

  require_file "${input_jsonl}"
  require_file "${diverse_query}"

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name}: split positives by median count"
  echo "  input-jsonl  : ${input_jsonl}"
  echo "  diverse-query: ${diverse_query}"
  echo "  output-dir   : ${output_dir}"

  python "${SPLIT_SCRIPT}" \
    --input-jsonl "${input_jsonl}" \
    --diverse-query "${diverse_query}" \
    --output-dir "${output_dir}" \
    --median-rounding ceil \
    "${extra_args[@]}"
}

require_file "${SPLIT_SCRIPT}"

run_split "msmarco-gen" "${MSMARCO_GEN_JSONL}" "${MSMARCO_GEN_DIVERSE_QUERY}" --split-count 7
run_split "msmarco-ori" "${MSMARCO_ORI_JSONL}" "${MSMARCO_ORI_DIVERSE_QUERY}"
run_split "dbpedia" "${DBPEDIA_JSONL}" "${DBPEDIA_DIVERSE_QUERY}"
run_split "touche2020" "${TOUCHE2020_JSONL}" "${TOUCHE2020_DIVERSE_QUERY}"
run_split "fiqa" "${FIQA_JSONL}" "${FIQA_DIVERSE_QUERY}"
run_split "scidocs" "${SCIDOCS_JSONL}" "${SCIDOCS_DIVERSE_QUERY}"
run_split "nfcorpus" "${NFCORPUS_JSONL}" "${NFCORPUS_DIVERSE_QUERY}"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done."
