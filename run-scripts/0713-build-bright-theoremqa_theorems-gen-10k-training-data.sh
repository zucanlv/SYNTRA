#!/usr/bin/env bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ustc/public_envs/DSA

PROJECT_ROOT="/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch"
RESULTS_ROOT="${PROJECT_ROOT}/Main_Pipeline/Results"
PREPROCESS_DIR="${PROJECT_ROOT}/Main_Pipeline/corpus_preprocess"
OUTPUT_DIR="/data/share/project/shared_datasets/DSA/syn_data/RQ2-reasoning-intensive-task"
PROMPT="Given a Math problem, retrieve relevant theorems that help answer the problem."

# Select only a completed theoremqa_theorems 10k run.  A completed run records
# this marker in its main log, so an in-progress generation is never consumed.
RESULT_DIR=""
while IFS= read -r candidate; do
  log_file="${candidate}/Logs/$(basename "${candidate}" | sed 's/^bright-documents-theoremqa_theorems_//' | sed 's/^/main_/').log"
  if [[ -f "${log_file}" ]] && rg -q "Pipeline execution finished successfully" "${log_file}" && compgen -G "${candidate}/Annotated_Main_*.json" > /dev/null; then
    RESULT_DIR="${candidate}"
    break
  fi
done < <(find "${RESULTS_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'bright-documents-theoremqa_theorems_*' -printf '%T@ %p\n' | sort -rn | awk '{print $2}')

if [[ -z "${RESULT_DIR}" ]]; then
  echo "No completed theoremqa_theorems run with Annotated_Main output was found." >&2
  exit 1
fi

ANNOTATED_FILE="$(find "${RESULT_DIR}" -maxdepth 1 -type f -name 'Annotated_Main_*.json' -print -quit)"
RUN_ID="${RESULT_DIR##*_}"
OUTPUT_STEM="bright-documents-theoremqa_theorems-${RUN_ID}-gen-10k"
TRAINING_DATA="${OUTPUT_DIR}/${OUTPUT_STEM}.jsonl"
FILTERED_TRAINING_DATA="${OUTPUT_DIR}/${OUTPUT_STEM}_exclude_other_pos_save_non_pos.jsonl"

mkdir -p "${OUTPUT_DIR}"
python "${PREPROCESS_DIR}/anno_queries_to_msmarco_syn.py" \
  --input "${ANNOTATED_FILE}" --output "${TRAINING_DATA}" --prompt "${PROMPT}"
python "${PREPROCESS_DIR}/4-29-exclude-other-pos-save-non-pos.py" \
  --input-jsonl "${TRAINING_DATA}" --diverse-query "${ANNOTATED_FILE}" --output "${FILTERED_TRAINING_DATA}"

printf '%s\n' "${TRAINING_DATA}" "${FILTERED_TRAINING_DATA}" > "${OUTPUT_DIR}/latest-theoremqa_theorems-gen-10k-training-data.paths"
echo "Completed run: ${RESULT_DIR}"
echo "Training data: ${TRAINING_DATA}"
echo "Filtered training data: ${FILTERED_TRAINING_DATA}"

