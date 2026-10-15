# SYNTRA

**Self-improving data synthesis enables scalable adaptation of embedding models**  
Zucan Lyu, Rui Tang, Jianlyu Chen, Peisheng Jin and Defu Lian

[![arXiv](https://img.shields.io/badge/arXiv-2610.14730-b31b1b.svg)](https://arxiv.org/abs/2610.14730)
[![Project page](https://img.shields.io/badge/Project-page-blue)](https://zucanlv.github.io/SYNTRA/)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20Data-SYNTRA-yellow)](https://huggingface.co/datasets/ZucanLyu/SYNTRA)

SYNTRA uses few-shot-guided self-refinement to adapt a shared agentic framework
for retrieval training-data synthesis.

This release preserves the research workflow: define a task and its few-shot
examples, configure the existing pipeline, refine instructions on a small sample,
then reuse those instructions for production and export training records.

## Paper and resources

- Paper: [arXiv:2610.14730](https://arxiv.org/abs/2610.14730) ([PDF](https://arxiv.org/pdf/2610.14730)). The arXiv version includes the Supplementary Information.
- [Supplementary Note 1 accompanying files](docs/supplementary/).
- [Project page](https://zucanlv.github.io/SYNTRA/) with [FAQ](https://zucanlv.github.io/SYNTRA/#faq).
- [Paper summary](docs/PAPER_SUMMARY.md) and [key results as CSV](paper/key_results.csv).
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

## Questions and contributions

Questions, bug reports and results on new retrieval tasks are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) and the [issue templates](https://github.com/zucanlv/SYNTRA/issues/new/choose).

## Citation

If you use SYNTRA, its code or its data, please cite the paper:

```bibtex
@misc{lyu2026syntra,
  title         = {Self-improving data synthesis enables scalable adaptation of embedding models},
  author        = {Lyu, Zucan and Tang, Rui and Chen, Jianlyu and Jin, Peisheng and Lian, Defu},
  year          = {2026},
  eprint        = {2610.14730},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR},
  doi           = {10.48550/arXiv.2610.14730},
  url           = {https://arxiv.org/abs/2610.14730}
}
```

## License

The original SYNTRA code and associated software documentation are released under
the [MIT License](LICENSE). Third-party code retains its upstream licenses and
attribution; see [third-party notices](THIRD_PARTY_NOTICES.md). The code license
does not relicense the manuscript PDFs, datasets or benchmark-derived text and
examples; those materials remain subject to their respective terms.
