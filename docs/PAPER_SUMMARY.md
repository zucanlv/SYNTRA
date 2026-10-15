# SYNTRA paper summary

Machine-readable summary of *Self-improving data synthesis enables scalable adaptation of embedding models* (Zucan Lyu, Rui Tang, Jianlyu Chen, Peisheng Jin and Defu Lian; [arXiv:2610.14730](https://arxiv.org/abs/2610.14730)). Numbers are taken verbatim from the paper; [paper/key_results.csv](../paper/key_results.csv) lists them with their source sections.

**Keywords:** data synthesis, synthetic training data, embedding models, embedding model fine-tuning, self-improvement, LLM agents, information retrieval, domain adaptation

## What is SYNTRA?

SYNTRA is a self-improving agentic framework for retrieval data synthesis. Given an unlabelled corpus and a few example query–document pairs, it generates synthetic training data (queries with positive and negative documents) for fine-tuning embedding models on a new retrieval domain or task.

## What do I need to provide?

An unlabelled document corpus and a small set of few-shot examples, each pairing a query with a relevant (positive) document. SYNTRA infers a task definition (query type, document type and relevance criteria) from these examples; users can review it or supply a task definition directly.

## What does SYNTRA output?

Training records in JSON Lines format. Each record contains a query, one or more positive documents and negative documents, with negatives labelled as hard or easy negatives by LLM relevance-label synthesis. Records can be used directly to fine-tune embedding models.

## What does “self-improving” mean here?

In a small-sample test mode, SYNTRA generates trial outputs, evaluates them against the task definition and the few-shot examples, and uses the feedback to revise its shared synthesis instructions for query generation and relevance labelling. The refined instructions are then fixed and reused for large-scale production. The self-improvement acts on the data-synthesis instructions, not on the weights of the LLM.

## Does self-improvement actually help?

Yes, on all six tasks tested. The largest gains were on NFCorpus (nDCG@10 24.83 → 36.01) and FiQA (28.87 → 35.53). Automatic self-improvement also gave higher retrieval scores than human calibration of the instructions on four of the six tasks; human calibration remained better on TheoremQA theorems.

## Which embedding models were trained with SYNTRA data?

A Qwen3-0.6B backbone with LoRA adaptation (main cross-task evaluation over 24 datasets in 15 domains), BGE-M3 and Qwen3-Embedding-8B (with and without existing labelled training sets), and a Qwen3-8B backbone (data-scaling experiment on MS MARCO).

## Can SYNTRA data help if I already have labelled training data?

In the paper, augmenting existing labelled training sets with SYNTRA-generated data improved both BGE-M3 and Qwen3-Embedding-8B on all three evaluated tasks (TheoremQA questions, SpartQA and WinoGrande). For tasks without labelled training sets (LeCaRDv2, EvidenceBench and SciRepEval Search), fine-tuning on SYNTRA data improved both models over their pretrained versions.

## Are synthetic queries as good as real user queries?

Under matched training conditions on the same 10,000 MS MARCO source documents, models trained with SYNTRA-generated queries scored higher than models trained with the original human (Bing search-log) queries: mean nDCG@10 42.87 vs 42.18 on BEIR and 55.51 vs 54.53 on NanoBEIR.

## How reliable are the LLM-generated relevance labels?

Against reference labels from seven human annotators on 150 query–document pairs, Qwen3-30B-A3B labels reached 91.3% binary agreement (Cohen’s κ = 0.719) and 68.7% three-class agreement (κ = 0.510). Most three-class disagreements were between hard and easy negatives.

## Does more synthetic data keep helping?

On MS MARCO, nDCG@10 increased at each of six evaluated scales, from 30.88 with 10,000 to 33.20 with 320,000 SYNTRA-generated training examples (a 32-fold range), with a fixed Qwen3-8B training recipe.

## How much human effort and compute does it need?

In the main experiments, human involvement was limited to task setup, typically no more than 10 minutes per task (retrospective estimate). Data production used Qwen3-30B-A3B on four NVIDIA GeForce RTX 5090 GPUs; most production batches of roughly 10,000 documents took about 2–10 hours, excluding calibration, indexing and model training.

## What are the known limitations?

Distinguishing hard from easy negatives is harder to label consistently than binary relevance; data scaling was tested on MS MARCO only; human calibration still outperformed automatic self-improvement on TheoremQA theorems; and SYNTRA is run separately for each retrieval task.

## Where are the code and data?

Code (MIT license): github.com/zucanlv/SYNTRA. Retrieval-training data exports: huggingface.co/datasets/ZucanLyu/SYNTRA. Paper: arXiv:2610.14730.

## Citation

See the [Citation section of the README](../README.md#citation) or [CITATION.cff](../CITATION.cff).
