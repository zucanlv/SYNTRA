#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat >&2 <<'USAGE'
Usage:
  bash run_flagembedding_custom_eval.sh <task_name> <dataset_dir> <model_path> <output_dir> [extra args...]
USAGE
  exit 2
fi

TASK_NAME="$1"
DATASET_DIR="$2"
MODEL_PATH="$3"
OUTPUT_DIR="$4"
shift 4

declare -A QUERY_INSTRUCTIONS=(
  [birco]='Given a complex information need, retrieve the most relevant text passages.'
  [birco_arguana]='Given a claim, retrieve arguments that refute the claim.'
  [birco_clinical_trial]='Given a clinical need, retrieve matching clinical trial records.'
  [birco_doris_mae]='Given a research information need, retrieve the most relevant scientific paper abstracts.'
  [birco_relic]='Given a literary analysis excerpt with a masked quotation, retrieve passages from the analyzed work that fit the gap and support the surrounding interpretation.'
  [birco_whatsthatbook]='Given a description of a book, retrieve the matching book passage.'
  [evidencebench]='Given a biomedical scientific hypothesis, retrieve relevant sentences from biomedical research papers that support, contradict, or qualify the hypothesis.'
  [financebench_open_source_subset]='Given a financial question, retrieve supporting evidence from SEC filings.'
  [kilt_retrieval___provenance_tasks]='Given a knowledge-intensive question, retrieve supporting Wikipedia provenance passages.'
  [kuake_qtr]='给定一个中文医疗查询，检索相关的医疗问题标题。'
  [lecard]='给定一段中文案情，检索相似或相关的判例。'
  [legalbench_rag]='Given a question about a legal contract or privacy policy, retrieve relevant contract or policy passages that directly answer it or provide decisive evidence.'
  [legalbench_rag_cuad]='Given a contract question, retrieve the relevant contract clause.'
  [legalcitebench_citation_retrieval]='Given a legal citation context, retrieve the cited case passage.'
  [scirepeval_search_tasks]='Given a scientific search query, retrieve the most relevant scientific paper abstracts.'
  [trialgpt_retrieval_benchmark]='Given a patient summary, retrieve matching clinical trials.'
)

if [[ -z "${QUERY_INSTRUCTIONS[$TASK_NAME]+set}" ]]; then
  echo "Unknown task name: $TASK_NAME" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON:-/data/share/project/public_envs/embedder_train_eval/bin/python}"
if [[ -n "${FLAGEMBEDDING_DEVICES:-}" ]]; then
  read -r -a DEVICES <<< "$FLAGEMBEDDING_DEVICES"
else
  DEVICES=(cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5)
fi
mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" -m FlagEmbedding.evaluation.custom \
  --eval_name "science_${TASK_NAME}" \
  --dataset_dir "$DATASET_DIR" \
  --splits test \
  --corpus_embd_save_dir "$OUTPUT_DIR/corpus_embd" \
  --output_dir "$OUTPUT_DIR/search_results" \
  --search_top_k 1000 \
  --cache_path /data/share/project/shared_datasets/.cache \
  --overwrite True \
  --k_values 10 100 \
  --eval_output_method markdown \
  --eval_output_path "$OUTPUT_DIR/eval_results.md" \
  --eval_metrics ndcg_at_10 recall_at_100 \
  --embedder_name_or_path "$MODEL_PATH" \
  --embedder_model_class decoder-only-base \
  --pooling_method last_token \
  --devices "${DEVICES[@]}" \
  --query_instruction_for_retrieval "${QUERY_INSTRUCTIONS[$TASK_NAME]}" \
  --query_instruction_format_for_retrieval 'Instruct: {}\nQuery: {}' \
  --trust_remote_code True \
  --use_bf16 True \
  --use_fp16 False \
  --embedder_batch_size 128 \
  --embedder_query_max_length 512 \
  --embedder_passage_max_length 512 \
  "$@"
