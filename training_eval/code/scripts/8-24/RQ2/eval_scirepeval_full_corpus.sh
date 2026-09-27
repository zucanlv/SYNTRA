#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <model_path> <output_dir> <bge-m3|qwen3-8b> [top_k]" >&2
  exit 2
fi

MODEL_PATH="$1"
OUTPUT_DIR="$2"
MODEL_FAMILY="$3"
TOP_K="${4:-10}"

export PATH="/data/share/project/public_envs/embedder_train_eval/bin:$PATH"
PYTHON_BIN="${EVALUATION_PYTHON:-/data/share/project/public_envs/embedder_train_eval/bin/python}"
EVALUATOR="${SCIREPEVAL_FULL_CORPUS_EVALUATOR:-/data/share/project/tr/data_synthesis_agent/code/scripts/8-24/RQ2/scirepeval_full_corpus.py}"
SCIREPEVAL_REPO="${SCIREPEVAL_REPO:-/data/share/project/tr/third_party/scirepeval}"
METADATA="${SCIREPEVAL_METADATA:-$SCIREPEVAL_REPO/local_data/search/evaluation.jsonl}"
QRELS="${SCIREPEVAL_QRELS:-$SCIREPEVAL_REPO/local_data/search/test/test_qrel.jsonl}"
PROMPT_FILE="${SCIREPEVAL_PROMPT_FILE:-$SCIREPEVAL_REPO/instr_prompts.json}"
DEFAULT_QWEN_QUERY_INSTRUCTION="Given a scientific literature search query, retrieve relevant scientific papers that directly address the specified research topic or satisfy the requested constraints."
QWEN_QUERY_INSTRUCTION="${QWEN_QUERY_INSTRUCTION-$DEFAULT_QWEN_QUERY_INSTRUCTION}"
QUERY_ARGS=()
# The Python process remaps CUDA_VISIBLE_DEVICES to cuda:0..N-1.  Pass those
# logical names to FlagEmbedding so it creates one encoding worker per GPU.
EVAL_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$EVAL_VISIBLE_DEVICES"
ENCODING_DEVICE_ARGS=()
for device_index in "${!VISIBLE_GPU_IDS[@]}"; do
  ENCODING_DEVICE_ARGS+=("cuda:$device_index")
done

case "$MODEL_FAMILY" in
  bge-m3) BACKEND=flagembedding-encoder ;;
  qwen3-8b)
    BACKEND=flagembedding-decoder
    if [[ -n "$QWEN_QUERY_INSTRUCTION" ]]; then
      QUERY_ARGS=(--query-instruction "$QWEN_QUERY_INSTRUCTION")
    fi
    ;;
  *) echo "Unsupported model family: $MODEL_FAMILY" >&2; exit 2 ;;
esac

[[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 2; }
[[ -f "$EVALUATOR" && -f "$METADATA" && -f "$QRELS" && -f "$PROMPT_FILE" ]] || {
  echo "Missing evaluator or SciRepEval data/prompt files" >&2; exit 2;
}
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$EVAL_VISIBLE_DEVICES" "$PYTHON_BIN" "$EVALUATOR" \
  --model-path "$MODEL_PATH" --scirepeval-repo "$SCIREPEVAL_REPO" \
  --metadata "$METADATA" --qrels "$QRELS" --prompt-file "$PROMPT_FILE" \
  --encoder-backend "$BACKEND" "${QUERY_ARGS[@]}" \
  --devices "${ENCODING_DEVICE_ARGS[@]}" --batch-size "${EVAL_BATCH_SIZE:-16}" \
  --retrieval-batch-size "${RETRIEVAL_BATCH_SIZE:-128}" --top-k "$TOP_K" \
  --device "${EVAL_DEVICE:-cuda:0}" \
  --result-output "$OUTPUT_DIR/scirepeval_full_corpus_results.json" \
  --run-output "$OUTPUT_DIR/scirepeval_full_corpus_top${TOP_K}.jsonl"
