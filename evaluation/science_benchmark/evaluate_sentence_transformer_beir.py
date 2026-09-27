from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from FlagEmbedding.evaluation.science_benchmark.beir_io import (
    read_beir_corpus,
    read_beir_queries,
    read_qrels,
)


def dcg(relevances: Iterable[int]) -> float:
    return sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def average_precision(
    retrieved: list[str], relevant: Mapping[str, int], k: int
) -> float:
    relevant_ids = {doc_id for doc_id, score in relevant.items() if score > 0}
    if not relevant_ids:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant_ids:
            hits += 1
            total += hits / rank
    return total / min(len(relevant_ids), k)


def reciprocal_rank(
    retrieved: list[str], relevant: Mapping[str, int], k: int
) -> float:
    relevant_ids = {doc_id for doc_id, score in relevant.items() if score > 0}
    for rank, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate(
    results: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
    ks: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    valid_query_ids = [query_id for query_id in qrels if qrels[query_id]]
    for k in ks:
        ndcgs: list[float] = []
        recalls: list[float] = []
        precisions: list[float] = []
        maps: list[float] = []
        mrrs: list[float] = []
        for query_id in valid_query_ids:
            relevant = qrels[query_id]
            ranked = sorted(
                results.get(query_id, {}),
                key=lambda doc_id: results[query_id][doc_id],
                reverse=True,
            )
            gains = [int(relevant.get(doc_id, 0)) for doc_id in ranked[:k]]
            ideal = sorted(
                [int(score) for score in relevant.values() if score > 0], reverse=True
            )[:k]
            ideal_dcg = dcg(ideal)
            ndcgs.append(dcg(gains) / ideal_dcg if ideal_dcg else 0.0)
            relevant_ids = {doc_id for doc_id, score in relevant.items() if score > 0}
            hits = sum(doc_id in relevant_ids for doc_id in ranked[:k])
            recalls.append(hits / len(relevant_ids) if relevant_ids else 0.0)
            precisions.append(hits / k if k else 0.0)
            maps.append(average_precision(ranked, relevant, k))
            mrrs.append(reciprocal_rank(ranked, relevant, k))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        metrics[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        metrics[f"precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
        metrics[f"map@{k}"] = float(np.mean(maps)) if maps else 0.0
        metrics[f"mrr@{k}"] = float(np.mean(mrrs)) if mrrs else 0.0
    return metrics


def search_topk(
    query_vectors: np.ndarray,
    corpus_vectors: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    top_k: int,
    query_batch_size: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    k = min(top_k, len(corpus_ids))
    for start in tqdm(
        range(0, len(query_ids), query_batch_size), desc="search", unit="batch"
    ):
        end = min(start + query_batch_size, len(query_ids))
        scores = query_vectors[start:end] @ corpus_vectors.T
        if k == len(corpus_ids):
            top_indices = np.argsort(-scores, axis=1)[:, :k]
        else:
            top_indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            partial_scores = np.take_along_axis(scores, top_indices, axis=1)
            ordering = np.argsort(-partial_scores, axis=1)
            top_indices = np.take_along_axis(top_indices, ordering, axis=1)
        for row, query_id in enumerate(query_ids[start:end]):
            results[query_id] = {
                corpus_ids[index]: float(scores[row, index])
                for index in top_indices[row]
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--metrics-k", default="1,3,5,10,20,100")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--prompt-name", default="query")
    parser.add_argument("--model-label", required=True)
    args = parser.parse_args()

    started = time.time()
    corpus = read_beir_corpus(args.data_dir)
    queries = read_beir_queries(args.data_dir)
    qrels = read_qrels(args.data_dir, split="test")
    query_ids = [query_id for query_id in queries if query_id in qrels]
    corpus_ids = list(corpus)
    if not query_ids or not corpus_ids:
        raise SystemExit("The converted task has no evaluable queries or corpus.")

    model = SentenceTransformer(
        args.model_path,
        device=args.device,
        model_kwargs={"torch_dtype": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = args.max_seq_length
    prompt_name = args.prompt_name if args.prompt_name in model.prompts else None
    corpus_vectors = model.encode(
        [corpus[doc_id] for doc_id in corpus_ids],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    query_vectors = model.encode(
        [queries[query_id] for query_id in query_ids],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        prompt_name=prompt_name,
    )
    corpus_vectors = np.asarray(corpus_vectors, dtype=np.float32)
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    results = search_topk(
        query_vectors,
        corpus_vectors,
        query_ids,
        corpus_ids,
        args.top_k,
        args.query_batch_size,
    )
    metrics = evaluate(
        results,
        {query_id: qrels[query_id] for query_id in query_ids},
        [int(k) for k in args.metrics_k.split(",") if k.strip()],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model_label,
        "model_path": args.model_path,
        "device": args.device,
        "prompt_name": prompt_name,
        "prompt": model.prompts.get(prompt_name, "") if prompt_name else "",
        "max_seq_length": model.max_seq_length,
        "embedding_dimension": int(corpus_vectors.shape[1]),
        "corpus": len(corpus_ids),
        "queries": len(query_ids),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_dir / "run.trec").open("w", encoding="utf-8") as handle:
        for query_id in query_ids:
            ranked = sorted(
                results[query_id],
                key=lambda doc_id: results[query_id][doc_id],
                reverse=True,
            )
            for rank, doc_id in enumerate(ranked, 1):
                handle.write(
                    f"{query_id} Q0 {doc_id} {rank} "
                    f"{results[query_id][doc_id]:.8f} {args.model_label}\n"
                )
    print(json.dumps({"metrics": metrics, "metadata": metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
