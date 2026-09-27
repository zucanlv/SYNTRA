# SYNTRA — code release staging

SYNTRA adapts a shared agentic retrieval-data synthesis framework to different tasks using few-shot-guided self-refinement.

This directory is the first curated code-only snapshot. It is not yet a validated portable installation or a published GitHub repository.

## Code map

- `Main_Pipeline/main.py`: current end-to-end synthesis entry point.
- `Main_Pipeline/`: few-shot/task definitions, instruction refinement, query synthesis, corpus indexing, relevance annotation, hard-negative mining, streaming and concurrency utilities.
- `Main_Pipeline/config/`: representative production, cross-task, scaling and ablation configurations.
- `Main_Pipeline/corpus_preprocess/` and `data_scripts/`: corpus adapters, training-data conversion, merging, sampling and dataset packaging code.
- `Main_Pipeline/test/`: existing regression and adapter checks. Some require external datasets, models or services.
- `run-scripts/`: synthesis and paper experiment launchers.
- `training_eval/code/`: shared evaluation implementation and retained paper training/evaluation recipes.
- `evaluation/`: SciRepEval full-corpus evaluation and MTEB/NanoBEIR support.
- `third_party/FlagEmbedding/`: source snapshot of the locally modified framework used for training/evaluation, plus the COIR evaluator. Its original license is retained.
- `human_evaluation/`: relevance annotation application and agreement-analysis code; annotation samples and responses are excluded.
- `docs/`: file-level provenance, exclusions, source versions, sanitization and validation records.

## Data boundary

Generated training data, raw corpora, retrieval indices, checkpoints, annotation records, cached outputs and run logs are not included. Datasets will be released separately on Hugging Face; no dataset repository has been created in this step. Task definitions and the small few-shot examples embedded in Python are retained as algorithm inputs. The only copied JSON file is the DeepSpeed configuration; JSON files under `docs/` are release metadata.

## Before running

The original experiment layout and parameters are preserved for traceability. Many scripts/configurations still contain lab-specific absolute data, model, environment and source paths. Adapt them before execution; do not run the historical launchers unchanged. Global LLM credentials can be supplied through `OPENAI_API_KEY`, and the endpoint through `OPENAI_BASE_URL`; copied YAML credential values have been cleared. Some standalone scripts still require their credential setup to be converted to environment variables.

The main entry point accepts `--config`. Run it from `Main_Pipeline` after configuring external corpus/model paths and the relevant Python dependencies. Refine instructions in test mode before a production run that reuses them. The historical `config-example.yaml` is a parameter reference, not a ready-to-run demo.

See `docs/RELEASE_NOTES.md` for the remaining packaging work and the limits of validation. No new project license has been selected; upstream licenses apply to retained third-party code.
