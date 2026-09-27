# FlagEmbedding evaluation adapters

This directory makes every benchmark in `benchmark.csv` evaluable with
FlagEmbedding by normalizing task data to a BEIR-style retrieval layout first.

## Target layout

Each converted task is written as:

```text
<task_dir>/
  corpus.jsonl
  queries.jsonl
  qrels/
    test.tsv
```

`corpus.jsonl` rows:

```json
{"_id": "doc-id", "title": "", "text": "document text"}
```

`queries.jsonl` rows:

```json
{"_id": "query-id", "text": "query text"}
```

`qrels/test.tsv`:

```text
query-id	corpus-id	score
```

## Evaluate with FlagEmbedding

```bash
/data/share/project/public_envs/embedder_train_eval/bin/python -m flagembedding_eval.evaluate_beir \
  --data-dir data/converted/scifact \
  --model-name-or-path BAAI/bge-base-en-v1.5 \
  --output-dir outputs/scifact
```

The evaluator imports `FlagAutoModel` from FlagEmbedding and computes retrieval
metrics from the resulting embeddings. It does not require MTEB support for the
task once the data is converted.

## Convert non-MTEB tasks

Use task presets where available:

```bash
/data/share/project/public_envs/embedder_train_eval/bin/python -m flagembedding_eval.prepare_tasks \
  --task KUAKE-QTR \
  --input-dir raw/CBLUE/CBLUEDatasets/KUAKE-QTR \
  --output-dir data/converted/kuake_qtr
```

For repositories that already provide separate corpus/query/qrels files but use non-BEIR names, pass them explicitly:

```bash
/data/share/project/public_envs/embedder_train_eval/bin/python -m flagembedding_eval.prepare_tasks \
  --task LOFin Benchmark \
  --corpus-file raw/corpus.tsv \
  --queries-file raw/queries.tsv \
  --qrels-file raw/qrels.tsv \
  --output-dir data/converted/lofin
```

For an arbitrary pairwise JSONL file, use the generic converter:

```bash
/data/share/project/public_envs/embedder_train_eval/bin/python -m flagembedding_eval.prepare_tasks \
  --task generic_jsonl \
  --input-file raw/task.jsonl \
  --output-dir data/converted/my_task \
  --query-field question \
  --doc-field evidence \
  --label-field label
```

HuggingFace-hosted datasets can be cached and converted directly:

```bash
/data/share/project/public_envs/embedder_train_eval/bin/python -m flagembedding_eval.prepare_tasks \
  --task LegalCiteBench \
  --hf-dataset legalcitebench/LegalCiteBench \
  --hf-split test \
  --output-dir data/converted/legalcitebench
```

## Supported non-MTEB adapter presets

- `KUAKE-QTR`
- `FinanceBench Open-Source Subset`
- `LOFin Benchmark`
- `LegalBench-RAG`
- `LegalBench-RAG CUAD`
- `LegalCiteBench Citation Retrieval`
- `ScholarQABench`
- `EvidenceBench`
- `KILT Retrieval / Provenance Tasks`
- `Open RAG Benchmark`
- `LeCaRD`
- `LeCaRDv2`
- `BIRCO`
- `TrialGPT Retrieval Benchmark`
- `SciRepEval Search Tasks`
- `generic_jsonl`

Some repositories have multiple historical schemas. The converters are written
to be defensive: they accept common field aliases and fail with a clear error
when a required query/document/evidence field cannot be inferred.
