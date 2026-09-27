# Selection and release notes

## Included

The currently deployed Main_Pipeline implementation and its local Python dependencies; RQ1–RQ4 and cross-task configurations; production configurations; reusable preprocessing; current evaluation; later experiment recipes; regression tests; human evaluation source; the actual modified FlagEmbedding training package and COIR evaluation code. File-level reasons and checksums are in file_manifest.json.

## Excluded

The entire old/, paper/, course_lab2/, workspace/, prompts/ output collections; editor/agent configuration; Git history; March/April training exploration; explicit debug branches; integrated demo and ad-hoc top-level test; early dated preprocessing utilities unless still useful; corpus_preprocess/before and Misc; generated case-study reports; DataFlow and Qwen demo checkouts; installed environments and caches; data, weights, indices, production releases, credentials, logs, annotation responses and archives. Qwen-Agent is an external dependency, not a copied full repository.

## Remaining packaging work

1. Replace lab-specific paths with shared CLI/environment configuration and provide a minimal runnable example.
2. Freeze dependencies separately for synthesis and training/evaluation, including GPU FAISS and the compatible Qwen-Agent APIs.
3. Reconcile overlapping historical experiment recipes into canonical paper-table commands. Retained RQ recipes are evidence and candidates, not a claim that every file produced a final reported result.
4. Publish external datasets on Hugging Face and connect loaders and human-evaluation sample preparation. Some tests intentionally need those external inputs.
5. Confirm project licensing and attribution for benchmark-derived few-shot examples and nested third-party COIR/BEIR code. Retained FlagEmbedding LICENSE does not by itself settle every nested dependency's origin.
6. Run GPU/API end-to-end reproduction after paths and environments are configured. This staging step does not launch training or paid model calls.

Only the new staging directory is modified. Original source and production data are untouched. No GitHub or Hugging Face upload is performed.

## Validation completed

Python syntax: 404 files passed. Shell syntax: 278 files passed. YAML parsing: 95 configurations passed. Core pipeline local imports are present. Two offline data merge/split regression tests passed. No generated data files, model weights, symlinks or known credential-token patterns were found. Source checksums matched during copying. See validation.json for details; this is not an end-to-end model reproduction. Environment versions and lab-path migration inventory are recorded separately.
