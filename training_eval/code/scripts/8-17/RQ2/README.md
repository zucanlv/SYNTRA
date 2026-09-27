# SciRepEval Search custom evaluation

This directory evaluates only the converted SciRepEval **Search** task with
`FlagEmbedding.evaluation.custom`. It is not a complete SciRepEval official score.

1. Download or locate `search/evaluation-00000-of-00001.parquet` from
   `allenai/scirepeval`.
2. Fill `MODEL_PATH`, `OUTPUT_DIR`, and `RAW_EVALUATION_PARQUET` in `run.sh`.
3. Run `bash run.sh`.

The conversion writes `dataset_custom/corpus.jsonl`,
`dataset_custom/test_queries.jsonl`, and `dataset_custom/test_qrels.jsonl` below
the output directory. The native custom evaluator writes its result artifacts
below the same output directory. Set `PYTHON` to override the evaluation Python
interpreter and `FLAGEMBEDDING_DEVICES` (for example, `cuda:0 cuda:1`) to choose
GPUs.


## Official SciRepEval evaluation for BGE-M3

`eval_scirepeval_official_bge_m3.sh` runs the original
`allenai/scirepeval/scirepeval.py` runner with the encoder-only default model
path and CLS pooling. It requires a cloned SciRepEval repository plus its
prepared Hugging Face datasets cache:

```bash
SCIREPEVAL_REPO=/path/to/scirepeval \
HF_DATASETS_CACHE=/path/to/huggingface/datasets/cache \
SCIREPEVAL_PYTHON=/path/to/scirepeval/python \
bash eval_scirepeval_official_bge_m3.sh /path/to/bge-m3 /path/to/output Search
```

Omit `Search` to run every standard task. This is the original benchmark path;
the BGE-M3 sparse and ColBERT representations are not evaluated.


## Qwen3-0.6B training

Use `train_qwen_0.6b.sh <output_dir> <train_data>` to train the same
Qwen3-0.6B decoder-only embedding configuration used by the `8-15` RQ2 scripts.
It writes the merged LoRA checkpoint below `<output_dir>/merged_model` for a
subsequent Qwen3-compatible SciRepEval evaluation run. Override GPU count or
training defaults with the environment variables printed by `--help`.


## Official SciRepEval evaluation for Qwen3-0.6B

Use `eval_scirepeval_official_qwen3_0.6b.sh` for the original SciRepEval
Qwen3 adapter. It invokes `scirepeval.py --instructor --model-type qwen3` with
the repository's `instr_prompts.json`; `Search` is optional and omitting it runs
the standard task configuration.

```bash
SCIREPEVAL_REPO=/path/to/scirepeval \
HF_DATASETS_CACHE=/path/to/huggingface/datasets/cache \
SCIREPEVAL_PYTHON=/path/to/scirepeval_env/bin/python \
bash eval_scirepeval_official_qwen3_0.6b.sh \
  /path/to/train-output/merged_model /path/to/output Search
```

The official Qwen3 adapter requires `transformers >= 4.51` and loads the model
through `SentenceTransformer`. To retain last-token pooling from a raw
`merged_model`, provide a SentenceTransformer-compatible model directory with a
last-token pooling module; otherwise SentenceTransformers may select its default
pooling for a plain Transformers checkpoint.
