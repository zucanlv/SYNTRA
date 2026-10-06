# Run the original SYNTRA workflow

This guide keeps the existing task dictionaries, YAML loader, self-refinement and
`python main.py` entry point. It uses the MS MARCO generation route recorded in
`config-0606-msmarco-10k-gen.yaml` and `0607-build-gen-touche-training-data.sh` as a
concrete reference. It is not a claim that one configuration reproduces every
paper result.

## 1. Environment and corpus

Follow [INSTALL.md](INSTALL.md). Work from `Main_Pipeline` for every pipeline
command: the original code mixes current-directory-relative outputs and
module-relative instruction paths.

```bash
cd Main_Pipeline
cp config/delivery/msmarco-from-corpus.yaml config/local.yaml
```

Edit `config/local.yaml` before running:

- Set `corpus_path` to your own **absolute** corpus path. Use a small, explicitly
  selected corpus first. `--test-mode` limits query generation to five sampled
  documents by default, but it does **not** limit corpus indexing or guarantee a
  particular number of API requests.
- Set `faiss_dir` and `instructions_store_dir` to separate, dedicated absolute
  output directories. Use a fresh instruction directory when task definitions,
  examples or models change; the original cache lookup does not validate all of
  those changes. Never point output paths at your source data or existing runs.
- Set `llm_model` to a model served by your endpoint. If different stages use
  different models, populate the existing `llm_stages` mapping.
- Set the endpoint using `OPENAI_BASE_URL` and provide `OPENAI_API_KEY` through
  your shell/secret manager. No keys should be written to the YAML: the pipeline
  copies configuration files into results. Never commit local configuration.
- Check embedding model, device IDs, corpus columns and production sample size.

The corpus reader already supports JSONL, CSV, TSV, Parquet and HF `save_to_disk`
directories. A JSONL record can be `{"id":"doc-1","text":"document body"}`.
An HF repository ID alone is not a corpus path; download/prepare the dataset
separately. Raw corpora and generated datasets are not shipped in this code repo.

The template preserves the reference's synthesis settings: one query per source
document, 20 retrieved candidates, four candidates per annotation call,
`multi_doc_with_doc_id`, `fewshot_positive` calibration with five maximum attempts,
and the original ablation/filter switches. Delivery changes are explicit:
lab paths/model endpoints are replaced by user inputs; saved intermediate inputs
are cleared; indexing starts from the supplied corpus; FAISS is CPU-configured;
concurrency is four rather than 128. Original preselected documents are not
distributed, so fresh-corpus selection is not an exact replay of that experiment.
The default production sample remains 10,000; reduce it explicitly for your first
production check. The model endpoint is intentionally not silently selected.

## 2. Task definition and few-shot examples

For a built-in task, retain its entry in `HighLevel_Def.py` and
`Few_Shot_Example.py`. For a new task, add the same lowercase key to both:

```python
# HighLevel_Def.py
"my_task": ("query type", "document type", "What makes a document relevant"),

# Few_Shot_Example.py
"my_task": [
    {"query": "example query", "Positive": "relevant document text"},
],
```

Set `task: my_task` in the local YAML. The exact field names above matter; optional
`Hard Negative` holds the negative document text. Supply complete example text,
not unresolved corpus IDs. Use examples representative of your task. With
`fewshot_positive`, positive examples guide calibration; do not assume supplied
negative examples are necessarily used by that calibration branch.

`HighLevel_Generator.py` is an optional existing utility for proposing a task
definition. The main pipeline does not automatically invoke it. Review its output
and put the accepted definition into `HighLevel_Def.py` before running.

## 3. Refine on a small sample

```bash
python ../scripts/check_delivery.py --config config/local.yaml --phase refine --check-imports
python main.py --config config/local.yaml --test-mode
```

The first command is read-only and makes no model calls. The second runs the real
pipeline, including LLM calls and indexing. The template keeps `test_mode: false`;
the CLI flag enables refinement for this invocation. This distinction matters:
the original `--test-mode` is a one-way switch, not an on/off override.

Inspect `Results/<task>_<timestamp>/`: generated queries, candidate annotations,
logs and refinement histories. The shared instruction directory receives final
`IdAttrRefine_*`, `QueryInstrRefine_*` and `AnnoInstrRefine_*` files. Inspect their
content and refinement pass/fail history before production. File existence alone
does not demonstrate good task adaptation. Existing `human_review: true` can be
used for interactive review during refinement.

## 4. Produce using the saved instructions

Keep the same task, examples, model choices and instruction directory. Decide the
production size explicitly in `production_sample.count`, then run:

```bash
python ../scripts/check_delivery.py --config config/local.yaml --phase production --check-imports
python main.py --config config/local.yaml
```

The pipeline loads final instructions from the shared store. Outputs are under
`Main_Pipeline/Results/<task>_<timestamp>/`, not next to the YAML. Avoid concurrent
launches for the same task within the same second; original run names have
second-level timestamps. The check covers this documented from-corpus route, not
all historical ablation or resume modes. Existing stage-4/stage-5 resume inputs
remain available in the original configuration reference.

## 5. Convert annotations to training records

Choose the **specific production run** you inspected. Do not automatically select
the newest file, which might belong to refinement or another experiment.

```bash
python corpus_preprocess/anno_queries_to_msmarco_syn.py \
  --input /absolute/path/to/your/run/Annotated_Main_TIMESTAMP.json \
  --output /absolute/path/to/your/export/train.jsonl \
  --prompt "Given a web search query, retrieve relevant passages that answer the query."
```

Use the actual `Annotated_*.json` filename in that run. This existing converter
uses `positive_doc_ids` and `hard_negative_doc_ids`, resolves candidate text, and
writes `prompt`, `query`, `pos`, `neg`. It drops records without positives or hard
negatives. An empty export is therefore possible; inspect annotation quality and
converter counts instead of assuming every generated query becomes a record.

The reference build script additionally applies:

```bash
python corpus_preprocess/4-29-exclude-other-pos-save-non-pos.py \
  --input-jsonl /absolute/path/to/your/export/train.jsonl \
  --diverse-query /absolute/path/to/your/run/DiverseQuery_Main_TIMESTAMP.json \
  --output /absolute/path/to/your/export/train_filtered.jsonl
```

Use the corresponding actual `DiverseQuery_*.json` file. Other experiments use
score-3-only or score-2-fallback converters; those are distinct data selection
policies, not interchangeable defaults. Keep exports outside the repository for
the planned Hugging Face dataset release.

## Research recipes

`run-scripts/` and `training_eval/code/scripts/` preserve experimental evidence.
They include lab paths, environment activation and multi-GPU settings. Read and
adapt a chosen recipe before executing it. Start with this guide for synthesis;
use a specific experiment's configuration and converters for paper reproduction.
