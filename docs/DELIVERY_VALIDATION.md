# Original-workflow delivery validation

Checked on 2026-10-06 against source baseline
`8507349462cf015ad5f7dd2756cb9e135323d07f`.

## Scope of the changes

The original `main.py`, task definitions, few-shot examples, prompts,
self-refinement logic and historical configurations are unchanged. The delivery
adds a workflow guide, observed dependency pins, a separately named configuration
template, read-only prerequisite checks and focused regression tests.

Two startup problems found during verification were repaired:

- Seven imported pipeline modules opened log files at import time. Two expected
  `Results/Logs` to exist, so even `main.py --help` failed in a fresh checkout.
  Standalone modules now initialize those file handlers only when executed
  directly; `main.py` retains its existing runtime logging setup. Function and
  class bodies in those seven files are unchanged (checked by AST comparison).
- The documented annotation converter indexed `Path.parents[5]` before parsing
  explicit input/output arguments. This failed for shallow checkout paths.
  Its historical lab default is now a literal path, leaving conversion behavior
  unchanged and allowing explicit paths at any checkout depth.

The new template was compared with `config-0606-msmarco-10k-gen.yaml`. Differences
are limited to input/output/model paths, per-stage endpoint/model placeholders,
fresh-corpus indexing instead of saved intermediate input, CPU FAISS device
settings, and concurrency reduced from 128 to four. See
[the workflow guide](ORIGINAL_WORKFLOW.md) for the distinction between this
starting route and exact experimental reproduction.

## Checks passed

Tests ran in a separate temporary source copy on Linux with Python 3.10.20 and
the existing research synthesis environment. Model downloads were disabled,
CUDA was hidden, and no real API credentials or generation calls were used.

| Check | Result |
| --- | --- |
| Delivery prerequisite checks | 7 tests passed, including malformed YAML, secret-free errors, read-only behavior and production instruction prerequisites |
| Original export commands | 1 test passed: annotation IDs resolve to positive/negative text, unusable records are dropped, and origin-positive filtering matches the documented recipe |
| Fresh-directory CLI startup | 1 test passed: `main.py --help` succeeds without credentials or precreated log directories, and creates no files in its working directory |
| Existing merge/split/score-based conversion tests | 6 tests passed |
| Pipeline imports through the delivery check | Passed with a temporary corpus/config and dummy endpoint; no model calls |
| Unedited template / missing production instructions | Rejected with actionable errors |
| Delivery template versus source configuration | Only the documented deployment changes found |

The published PyPI wheel for Qwen-Agent 0.0.34 was also checked: it contains
`qwen_agent.utils.parallel_executor`, and that module matched the installed copy.

To repeat the nine delivery tests after installing the synthesis dependencies,
run from the repository root:

```bash
python -m unittest discover -s tests -v
```

The six selected existing tests can be repeated with:

```bash
python -m unittest discover -s Main_Pipeline/test/test_corpus_preprocess -p test_merge_pipeline_json.py
python -m unittest discover -s Main_Pipeline/test/test_corpus_preprocess -p test_split_queries_by_position.py
python -m unittest discover -s Main_Pipeline/test/test_corpus_preprocess -p test_anno_queries_to_msmarco_syn_score3_fallback_score2.py
python -m unittest discover -s Main_Pipeline/test/test_corpus_preprocess -p test_anno_queries_to_msmarco_syn_score3_pos.py
```

## Remaining verification

This was a fresh source copy using an existing environment, **not a clean
dependency installation**. The suggested CPU FAISS build has not been exercised
in a newly installed environment. Full indexing, live LLM refinement, production
synthesis, training and benchmark reproduction were not run for this delivery.
Passing prerequisite checks does not establish instruction quality or model
compatibility at a user's endpoint. Inspect a small real run before scaling up.

Temporary test records are synthetic fixtures, created during tests. Production
datasets, indices, checkpoints and credentials were not added to this repository.
