# Third-party notices

The root MIT license applies to original SYNTRA code and associated software
documentation. The components below retain their upstream terms and ownership.
Existing source-file copyright and attribution notices are preserved.

| Component | Included code | License and notices |
| --- | --- | --- |
| FlagEmbedding | `third_party/FlagEmbedding/`, except separately licensed components below | [Retained upstream MIT license](third_party/FlagEmbedding/LICENSE), copyright 2022 staoxiao |
| AIR-Bench | Adapted evaluation arguments in `evaluation/mmteb/arguments.py`, `evaluation/mmteb/qwen3/arguments.py`, `training_eval/code/arguments.py`; adapted files identified in `third_party/FlagEmbedding/FlagEmbedding/abc/evaluation/` | [Upstream MIT license](licenses/AIR-Bench-MIT.txt), copyright 2024 AIR-Bench; source version [0.1.0](https://github.com/AIR-Bench/AIR-Bench/tree/0.1.0) |
| COIR | `third_party/FlagEmbedding/research/BGE_Coder/evaluation/coir_eval/coir/` | [Upstream Apache 2.0 license](licenses/COIR-Apache-2.0.txt); [source repository](https://github.com/CoIR-team/coir) |
| BEIR | The nested `coir/beir/` package above; BEIR-derived custom metrics identified in `third_party/FlagEmbedding/FlagEmbedding/abc/evaluation/utils.py` | [Upstream Apache 2.0 license](licenses/BEIR-Apache-2.0.txt) and [original notices](licenses/BEIR-NOTICE.txt); source revision cited in the code: [f062f038](https://github.com/beir-cellar/beir/tree/f062f038c4bfd19a8ca942a9910b1e0d218759d4) |
| Sentence Transformers | Copied or adapted multiprocessing methods identified by source links in FlagEmbedding's `abc/inference/AbsEmbedder.py`, `abc/inference/AbsReranker.py`, `inference/embedder/decoder_only/icl.py` and `inference/embedder/encoder_only/m3.py` | [Upstream Apache 2.0 license](licenses/Sentence-Transformers-Apache-2.0.txt) and [original notices](licenses/Sentence-Transformers-NOTICE.txt); source revision cited in the code: [1802076d](https://github.com/huggingface/sentence-transformers/tree/1802076d4eae42ff0a5629e1b04e75785d4e193b) |
| Transformers-derived MiniCPM and Gemma code | Files with Apache headers in `third_party/FlagEmbedding/FlagEmbedding/finetune/reranker/decoder_only/layerwise/` and `third_party/FlagEmbedding/FlagEmbedding/inference/reranker/decoder_only/models/` | [Upstream Apache 2.0 license](licenses/Transformers-Apache-2.0.txt); original EleutherAI, Google and Hugging Face copyright headers remain in the files |
| ChemDataExtractor text normalization | `third_party/FlagEmbedding/FlagEmbedding/evaluation/mkqa/utils/normalize_text.py` | The original MIT license and copyright 2016 Matt Swain are retained in the file header |
| KaTeX | `human_evaluation/web/vendor/` | [Retained upstream MIT license](human_evaluation/web/vendor/LICENSE.katex.txt), copyright 2013–2020 Khan Academy and other contributors |

The FlagEmbedding directory is a locally modified research snapshot, with source
revisions recorded in `docs/source_versions.json`. Copied and adapted methods
remain subject to their upstream terms; the root MIT license does not replace
those terms.

Dataset text and benchmark-derived few-shot examples are not covered by the code
license. This includes example records in `Main_Pipeline/Few_Shot_Example.py` and
the accompanying Supplementary Note 1 JSON files. The manuscript PDFs in `paper/`
are also outside the code license. See the benchmark sources and the public data
archive for the terms applicable to data and examples.
