# SYNTRA

SYNTRA uses few-shot-guided self-refinement to adapt a shared agentic framework
for retrieval training-data synthesis.

This release preserves the research workflow: define a task and its few-shot
examples, configure the existing pipeline, refine instructions on a small sample,
then reuse those instructions for production and export training records.

## Paper and resources

- [Manuscript PDF](paper/SYNTRA.pdf) (draft, 6 October 2026).
- [Project page](https://zucanlv.github.io/SYNTRA/).
- [Public data archive](https://huggingface.co/datasets/ZucanLyu/SYNTRA).

## Start here

1. [Install the synthesis environment](docs/INSTALL.md).
2. [Run the original workflow](docs/ORIGINAL_WORKFLOW.md).
3. Copy `Main_Pipeline/config/delivery/msmarco-from-corpus.yaml` to a local
   configuration and replace its marked paths and served-model name.
4. From `Main_Pipeline`, check the configuration and run refinement:

```bash
python ../scripts/check_delivery.py --config config/local.yaml --phase refine --check-imports
python main.py --config config/local.yaml --test-mode
```

Read the workflow guide before running these commands. `--test-mode` makes real
model calls and still indexes the supplied corpus; start with a small corpus.
Review the results and follow the guide for production and training-data export.
API keys belong in the environment, not configuration files or commits.

The template uses the original MS MARCO pipeline settings with explicit deployment
adjustments. Historical experiment configurations and default algorithms are
preserved. This is not an automatic new-task API: custom tasks still require
entries in `HighLevel_Def.py` and `Few_Shot_Example.py`, as shown in the guide.

## Repository map

| Directory | Purpose |
| --- | --- |
| `Main_Pipeline/` | Original synthesis pipeline and task definitions |
| `Main_Pipeline/config/delivery/` | Starting configuration for the documented original workflow |
| `Main_Pipeline/config/` | Preserved experiment configurations |
| `Main_Pipeline/corpus_preprocess/`, `data_scripts/` | Corpus preparation, conversion, merging and sampling |
| `run-scripts/`, `training_eval/` | Historical generation, training and evaluation recipes; adapt lab paths before use |
| `evaluation/` | SciRepEval and MTEB/NanoBEIR support |
| `third_party/FlagEmbedding/` | Locally modified research framework snapshot and retained upstream license |
| `human_evaluation/` | Human relevance evaluation and agreement-analysis code |
| `tests/`, `Main_Pipeline/test/` | Delivery checks and existing regression/adapter tests |
| `docs/` | Usage, provenance, observed environments and validation |

## Data and validation

Raw corpora, generated datasets, checkpoints, indices, annotation responses and run
logs are excluded from this code repository. Retrieval-training data exports are
publicly available in the [SYNTRA data archive](https://huggingface.co/datasets/ZucanLyu/SYNTRA).
The archive includes synthetic-query data, original-query controls and historical
experimental variants; consult its manifest and dataset card before selecting
files. Source benchmark text remains subject to its original terms. The embedded
few-shot examples remain part of the task configuration.

See [delivery validation](docs/DELIVERY_VALIDATION.md) for the checks actually run.
The `file_manifest.json` and original `validation.json` describe the initial
September 27 source snapshot; later delivery changes are tracked by Git and the
new validation note. They are not a current dependency lockfile or live checksum
inventory.

The main project license is still to be selected. Retained third-party license
files apply to their corresponding code; the repository does not yet declare a
blanket license for all contents.
