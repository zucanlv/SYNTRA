from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
from tqdm import tqdm

from .beir_io import read_beir_corpus, read_beir_queries, read_qrels


def load_flag_model(
    model_name_or_path: str,
    use_fp16: bool = True,
    query_instruction: str | None = None,
    model_class: str | None = None,
    query_instruction_format: str | None = None,
):
    try:
        from FlagEmbedding import FlagAutoModel
    except ImportError as exc:
        raise SystemExit(
            "FlagEmbedding is not installed. Install it first, for example:\n"
            "  pip install -U FlagEmbedding\n"
            "or run from the FlagEmbedding repository environment."
        ) from exc

    kwargs = {"use_fp16": use_fp16}
    if query_instruction:
        kwargs["query_instruction_for_retrieval"] = query_instruction
    if model_class:
        kwargs["model_class"] = model_class
    if query_instruction_format:
        kwargs["query_instruction_format"] = query_instruction_format
    return FlagAutoModel.from_finetuned(model_name_or_path, **kwargs)


def encode_texts(model, texts: List[str], batch_size: int, mode: str) -> np.ndarray:
    if mode == "query" and hasattr(model, "encode_queries"):
        vectors = model.encode_queries(texts, batch_size=batch_size)
    elif mode == "corpus" and hasattr(model, "encode_corpus"):
        vectors = model.encode_corpus(texts, batch_size=batch_size)
    else:
        vectors = model.encode(texts, batch_size=batch_size)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={vectors.shape}")
    return vectors


def normalize(vectors: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    denom[denom == 0] = 1
    return vectors / denom


def batched_topk(
    query_vectors: np.ndarray,
    corpus_vectors: np.ndarray,
    query_ids: List[str],
    corpus_ids: List[str],
    top_k: int,
    query_batch_size: int,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    k = min(top_k, len(corpus_ids))
    for start in tqdm(range(0, len(query_ids), query_batch_size), desc="search"):
        end = min(start + query_batch_size, len(query_ids))
        scores = query_vectors[start:end] @ corpus_vectors.T
        if k == len(corpus_ids):
            top_idx = np.argsort(-scores, axis=1)[:, :k]
        else:
            top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            row_scores = np.take_along_axis(scores, top_idx, axis=1)
            order = np.argsort(-row_scores, axis=1)
            top_idx = np.take_along_axis(top_idx, order, axis=1)
        for row, qid in enumerate(query_ids[start:end]):
            results[qid] = {corpus_ids[idx]: float(scores[row, idx]) for idx in top_idx[row]}
    return results


def dcg(relevances: Iterable[int]) -> float:
    return sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def average_precision(retrieved: List[str], relevant: Mapping[str, int], k: int) -> float:
    hits = 0
    score = 0.0
    rel_ids = {docid for docid, rel in relevant.items() if rel > 0}
    if not rel_ids:
        return 0.0
    for rank, docid in enumerate(retrieved[:k], 1):
        if docid in rel_ids:
            hits += 1
            score += hits / rank
    return score / min(len(rel_ids), k)


def reciprocal_rank(retrieved: List[str], relevant: Mapping[str, int], k: int) -> float:
    rel_ids = {docid for docid, rel in relevant.items() if rel > 0}
    for rank, docid in enumerate(retrieved[:k], 1):
        if docid in rel_ids:
            return 1.0 / rank
    return 0.0


def evaluate(results: Mapping[str, Mapping[str, float]], qrels: Mapping[str, Mapping[str, int]], ks: List[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    valid_qids = [qid for qid in qrels if qrels[qid]]
    for k in ks:
        ndcgs: list[float] = []
        recalls: list[float] = []
        precisions: list[float] = []
        maps: list[float] = []
        mrrs: list[float] = []
        for qid in valid_qids:
            relevant = qrels[qid]
            retrieved = sorted(results.get(qid, {}), key=lambda docid: results[qid][docid], reverse=True)
            gains = [int(relevant.get(docid, 0)) for docid in retrieved[:k]]
            ideal = sorted([int(rel) for rel in relevant.values() if rel > 0], reverse=True)[:k]
            ideal_dcg = dcg(ideal)
            ndcgs.append(dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0)
            rel_ids = {docid for docid, rel in relevant.items() if rel > 0}
            hits = sum(1 for docid in retrieved[:k] if docid in rel_ids)
            recalls.append(hits / len(rel_ids) if rel_ids else 0.0)
            precisions.append(hits / k if k else 0.0)
            maps.append(average_precision(retrieved, relevant, k))
            mrrs.append(reciprocal_rank(retrieved, relevant, k))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        metrics[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        metrics[f"precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
        metrics[f"map@{k}"] = float(np.mean(maps)) if maps else 0.0
        metrics[f"mrr@{k}"] = float(np.mean(mrrs)) if mrrs else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BEIR-style retrieval task with a FlagEmbedding model.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--metrics-k", default="1,3,5,10,20,100")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--query-instruction", default=None)
    parser.add_argument(
        "--query-instruction-format",
        default="Instruct: {}\\nQuery: {}",
        help="Template for query instructions; use the Qwen training template by default.",
    )
    parser.add_argument("--model-class", default=None, help="Optional FlagEmbedding model class, e.g. decoder-only-base for local Qwen snapshots.")
    args = parser.parse_args()

    corpus = read_beir_corpus(args.data_dir)
    queries = read_beir_queries(args.data_dir)
    qrels = read_qrels(args.data_dir, split=args.split)
    query_ids = [qid for qid in queries if qid in qrels]
    corpus_ids = list(corpus)
    if not query_ids:
        raise SystemExit("No queries with qrels were found.")
    if not corpus_ids:
        raise SystemExit("Corpus is empty.")

    model = load_flag_model(
        args.model_name_or_path,
        use_fp16=not args.no_fp16,
        query_instruction=args.query_instruction,
        model_class=args.model_class,
        query_instruction_format=args.query_instruction_format,
    )
    corpus_vectors = encode_texts(model, [corpus[docid] for docid in corpus_ids], args.batch_size, mode="corpus")
    query_vectors = encode_texts(model, [queries[qid] for qid in query_ids], args.batch_size, mode="query")
    if not args.no_normalize:
        corpus_vectors = normalize(corpus_vectors)
        query_vectors = normalize(query_vectors)

    results = batched_topk(query_vectors, corpus_vectors, query_ids, corpus_ids, args.top_k, args.query_batch_size)
    ks = [int(k) for k in args.metrics_k.split(",") if k.strip()]
    eval_qrels = {qid: qrels[qid] for qid in query_ids}
    metrics = evaluate(results, eval_qrels, ks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with (args.output_dir / "run.trec").open("w", encoding="utf-8") as f:
        for qid in query_ids:
            ranked = sorted(results[qid], key=lambda docid: results[qid][docid], reverse=True)
            for rank, docid in enumerate(ranked, 1):
                f.write(f"{qid} Q0 {docid} {rank} {results[qid][docid]:.8f} FlagEmbedding\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
