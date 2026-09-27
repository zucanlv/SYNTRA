#!/usr/bin/env python3
"""Evaluate SciRepEval Search by retrieving from the full evaluation corpus.

The official Search evaluation reranks ten judged candidates per query. This
script instead retrieves the top-k documents from the union of all candidate
papers in the evaluation split. Since relevance judgements remain limited to
the original ten candidates, unjudged corpus documents are treated as
non-relevant by pytrec_eval. The resulting metrics are diagnostic and are not
directly comparable to the official candidate-pool score.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger("scirepeval_full_corpus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scirepeval-repo", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--result-output", required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--embedding-output", default=None)
    parser.add_argument(
        "--encoder-backend",
        choices=(
            "official-qwen3",
            "flagembedding-decoder",
            "flagembedding-encoder",
        ),
        default="official-qwen3",
        help=(
            "Encoding implementation. Use flagembedding-decoder for raw merged "
            "FlagEmbedding checkpoints. Use flagembedding-decoder for Qwen "
            "decoder-only models and flagembedding-encoder for BGE-M3."
        ),
    )
    parser.add_argument(
        "--query-instruction",
        default=None,
        help=(
            "Instruction prepended only to FlagEmbedding queries. It must match "
            "the instruction used during training."
        ),
    )
    parser.add_argument(
        "--query-instruction-format",
        default="Instruct: {}\nQuery: {}",
        help="Template applied to --query-instruction and each query.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--retrieval-batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help=(
            "One or more logical devices for FlagEmbedding encoding, e.g. "
            "--devices cuda:0 cuda:1. Defaults to --device."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def load_metadata(path: str) -> Tuple[OrderedDict, List[str], int]:
    query_texts: OrderedDict[str, str] = OrderedDict()
    corpus_ids: List[str] = []
    seen_corpus_ids = set()
    candidate_pairs = 0

    for row in read_jsonl(path):
        query_id = str(row["doc_id"])
        query_texts[query_id] = str(row["query"])
        for candidate in row["candidates"]:
            candidate_pairs += 1
            doc_id = str(candidate["doc_id"])
            if doc_id not in seen_corpus_ids:
                seen_corpus_ids.add(doc_id)
                corpus_ids.append(doc_id)

    return query_texts, corpus_ids, candidate_pairs


def load_qrels(path: str) -> Tuple[OrderedDict, int]:
    qrels: OrderedDict[str, Dict[str, int]] = OrderedDict()
    pair_count = 0
    for row in read_jsonl(path):
        query_id = str(row["query_id"])
        doc_id = str(row["cand_id"])
        qrels.setdefault(query_id, {})[doc_id] = int(row["score"])
        pair_count += 1
    return qrels, pair_count


def flagembedding_model_options(
    backend: str,
    devices: str | List[str],
    batch_size: int,
    query_instruction: str | None = None,
    query_instruction_format: str = "Instruct: {}\nQuery: {}",
) -> dict:
    """Return FlagEmbedding settings matching the corresponding trainer."""
    common = {
        "normalize_embeddings": True,
        "use_fp16": False,
        "devices": devices,
        "trust_remote_code": True,
        "batch_size": batch_size,
        "query_max_length": 512,
        "passage_max_length": 512,
    }
    if backend == "flagembedding-decoder":
        return {
            **common,
            "model_class": "decoder-only-base",
            "pooling_method": "last_token",
            "query_instruction_for_retrieval": query_instruction,
            "query_instruction_format": query_instruction_format,
        }
    if backend == "flagembedding-encoder":
        return {
            **common,
            "model_class": "encoder-only-base",
            "pooling_method": "cls",
            # BGE-M3 is trained here without a query instruction.
            "query_instruction_for_retrieval": "",
        }
    raise ValueError(f"Unsupported FlagEmbedding backend: {backend}")


def load_embedding_texts(
    metadata_path: str,
    separator_token: str,
) -> Tuple[OrderedDict, OrderedDict]:
    """Build query and corpus texts exactly like SciRepEval's IRDataset."""
    query_texts: OrderedDict[str, str] = OrderedDict()
    corpus_texts: OrderedDict[str, str] = OrderedDict()
    fields = ("title", "abstract", "venue", "year")

    for row in read_jsonl(metadata_path):
        query_texts[str(row["doc_id"])] = str(row["query"])
        for candidate in row["candidates"]:
            doc_id = str(candidate["doc_id"])
            if doc_id in corpus_texts:
                continue
            values = [str(candidate[field]) for field in fields if candidate.get(field)]
            corpus_texts[doc_id] = f" {separator_token} ".join(values).strip()

    return query_texts, corpus_texts


def generate_flagembedding_embeddings(
    backend: str,
    model_path: str,
    metadata_path: str,
    batch_size: int,
    devices: str | List[str],
    query_instruction: str | None = None,
    query_instruction_format: str = "Instruct: {}\nQuery: {}",
) -> Mapping[str, np.ndarray]:
    """Encode a FlagEmbedding checkpoint with its explicit trained pooling."""
    from FlagEmbedding import FlagAutoModel

    options = flagembedding_model_options(
        backend,
        devices=devices,
        batch_size=batch_size,
        query_instruction=query_instruction,
        query_instruction_format=query_instruction_format,
    )
    LOGGER.info("Loading %s model with options: %s", backend, options)
    model = FlagAutoModel.from_finetuned(model_path, **options)

    separator_token = model.tokenizer.eos_token
    if not separator_token:
        raise RuntimeError("The FlagEmbedding tokenizer does not define an EOS token")
    query_texts, corpus_texts = load_embedding_texts(metadata_path, separator_token)

    query_ids = list(query_texts)
    corpus_ids = list(corpus_texts)
    query_embeddings = model.encode_queries(
        list(query_texts.values()),
        batch_size=batch_size,
        max_length=options["query_max_length"],
        convert_to_numpy=True,
    )
    corpus_embeddings = model.encode_corpus(
        list(corpus_texts.values()),
        batch_size=batch_size,
        max_length=options["passage_max_length"],
        convert_to_numpy=True,
    )

    query_norms = np.linalg.norm(query_embeddings, axis=1)
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)
    if not np.allclose(query_norms, 1.0, atol=1e-4):
        raise RuntimeError("FlagEmbedding query embeddings are not L2-normalized")
    if not np.allclose(corpus_norms, 1.0, atol=1e-4):
        raise RuntimeError("FlagEmbedding corpus embeddings are not L2-normalized")

    embeddings = {
        query_id: embedding
        for query_id, embedding in zip(query_ids, query_embeddings)
    }
    embeddings.update(
        {
            corpus_id: embedding
            for corpus_id, embedding in zip(corpus_ids, corpus_embeddings)
        }
    )
    LOGGER.info(
        "Generated %d normalized embeddings with %s", len(embeddings), backend
    )
    # Explicitly terminate one worker per GPU before retrieval/scoring.
    model.stop_self_pool()
    return embeddings


def generate_official_embeddings(
    model_path: str,
    scirepeval_repo: str,
    metadata_path: str,
    qrels_path: str,
    prompt_file: str,
    batch_size: int,
    embedding_output: str | None,
) -> Mapping[str, np.ndarray]:
    repo_path = str(Path(scirepeval_repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from evaluation.evaluator import IREvaluator
    from evaluation.instructor_new import Qwen3Model, load_prompts_from_file

    prompts = load_prompts_from_file(prompt_file, "blank")
    model = Qwen3Model(model_path, prompts)
    model.task_id = {"query": "[QRY]", "candidates": "[PRX]"}

    evaluator = IREvaluator(
        name="Search full corpus",
        meta_dataset=metadata_path,
        test_dataset=str(Path(qrels_path).parent),
        model=model,
        metrics=("ndcg_cut_10",),
        batch_size=batch_size,
        fields=["title", "abstract", "venue", "year"],
    )
    return evaluator.generate_embeddings(embedding_output)


def validate_embeddings(
    embeddings: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
) -> None:
    missing_queries = [query_id for query_id in query_ids if query_id not in embeddings]
    missing_docs = [doc_id for doc_id in corpus_ids if doc_id not in embeddings]
    if missing_queries or missing_docs:
        raise RuntimeError(
            "Embedding generation was incomplete: "
            f"missing_queries={len(missing_queries)}, missing_docs={len(missing_docs)}"
        )


def retrieve_full_corpus(
    embeddings: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    top_k: int,
    batch_size: int,
    device: str,
) -> OrderedDict:
    import torch

    if top_k <= 0 or top_k > len(corpus_ids):
        raise ValueError(f"top_k must be in [1, {len(corpus_ids)}], got {top_k}")

    corpus_array = np.stack([embeddings[doc_id] for doc_id in corpus_ids]).astype(
        np.float32, copy=False
    )
    corpus_tensor = torch.from_numpy(corpus_array).to(device)
    corpus_squared_norm = (corpus_tensor * corpus_tensor).sum(dim=1)
    corpus_transposed = corpus_tensor.transpose(0, 1)

    run: OrderedDict[str, Dict[str, float]] = OrderedDict()
    with torch.inference_mode():
        for start in range(0, len(query_ids), batch_size):
            batch_ids = query_ids[start : start + batch_size]
            query_array = np.stack([embeddings[query_id] for query_id in batch_ids]).astype(
                np.float32, copy=False
            )
            query_tensor = torch.from_numpy(query_array).to(device)

            # Squared Euclidean distance has the same ordering as Euclidean
            # distance and avoids an unnecessary square root over the full matrix.
            query_squared_norm = (query_tensor * query_tensor).sum(dim=1, keepdim=True)
            distances_squared = (
                query_squared_norm
                + corpus_squared_norm.unsqueeze(0)
                - 2.0 * query_tensor.matmul(corpus_transposed)
            ).clamp_min_(0.0)
            top_scores, top_indices = torch.topk(
                -distances_squared, k=top_k, dim=1, largest=True, sorted=True
            )

            top_scores = top_scores.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            for row_index, query_id in enumerate(batch_ids):
                run[query_id] = OrderedDict(
                    (
                        corpus_ids[int(doc_index)],
                        float(top_scores[row_index, rank]),
                    )
                    for rank, doc_index in enumerate(top_indices[row_index])
                )

            del query_tensor, distances_squared, top_scores, top_indices

    return run


def compute_metrics(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    top_k: int,
) -> Dict[str, float]:
    import pytrec_eval

    metric_names = {f"ndcg_cut_{top_k}", f"recall_{top_k}"}
    evaluator = pytrec_eval.RelevanceEvaluator(dict(qrels), metric_names)
    per_query = evaluator.evaluate(dict(run))

    metrics = {}
    for metric_name in sorted(metric_names):
        values = [query_metrics[metric_name] for query_metrics in per_query.values()]
        aggregate = pytrec_eval.compute_aggregated_measure(metric_name, values)
        metrics[metric_name] = float(np.round(100.0 * aggregate, 2))

    judged = 0
    positive_hits = 0
    retrieved = 0
    for query_id, results in run.items():
        query_qrels = qrels[query_id]
        query_has_positive = False
        for doc_id in results:
            retrieved += 1
            if doc_id in query_qrels:
                judged += 1
                if query_qrels[doc_id] > 0:
                    query_has_positive = True
        positive_hits += int(query_has_positive)

    metrics[f"judged_at_{top_k}"] = float(np.round(100.0 * judged / retrieved, 2))
    metrics[f"positive_hit_rate_at_{top_k}"] = float(
        np.round(100.0 * positive_hits / len(run), 2)
    )
    return metrics


def write_run(
    path: str,
    run: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
    query_texts: Mapping[str, str],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for query_id, results in run.items():
            query_qrels = qrels[query_id]
            row = {
                "query_id": query_id,
                "query": query_texts[query_id],
                "results": [
                    {
                        "rank": rank,
                        "doc_id": doc_id,
                        "retrieval_score": score,
                        "judged": doc_id in query_qrels,
                        "relevance": query_qrels.get(doc_id, 0),
                    }
                    for rank, (doc_id, score) in enumerate(results.items(), 1)
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    query_texts, corpus_ids, metadata_pairs = load_metadata(args.metadata)
    qrels, qrel_pairs = load_qrels(args.qrels)
    query_ids = list(qrels)

    missing_query_texts = [query_id for query_id in query_ids if query_id not in query_texts]
    if missing_query_texts:
        raise RuntimeError(f"Missing metadata for {len(missing_query_texts)} qrel queries")

    judged_counts = {len(query_qrels) for query_qrels in qrels.values()}
    LOGGER.info(
        "Loaded %d metadata queries, %d evaluated queries, %d unique corpus documents",
        len(query_texts),
        len(query_ids),
        len(corpus_ids),
    )

    if args.encoder_backend.startswith("flagembedding-"):
        if args.embedding_output:
            raise ValueError(
                "--embedding-output is not supported by FlagEmbedding backends"
            )
        embeddings = generate_flagembedding_embeddings(
            backend=args.encoder_backend,
            model_path=args.model_path,
            metadata_path=args.metadata,
            batch_size=args.batch_size,
            devices=args.devices if args.devices else args.device,
            query_instruction=args.query_instruction,
            query_instruction_format=args.query_instruction_format,
        )
    else:
        embeddings = generate_official_embeddings(
            model_path=args.model_path,
            scirepeval_repo=args.scirepeval_repo,
            metadata_path=args.metadata,
            qrels_path=args.qrels,
            prompt_file=args.prompt_file,
            batch_size=args.batch_size,
            embedding_output=args.embedding_output,
        )
    validate_embeddings(embeddings, query_ids, corpus_ids)

    run = retrieve_full_corpus(
        embeddings=embeddings,
        query_ids=query_ids,
        corpus_ids=corpus_ids,
        top_k=args.top_k,
        batch_size=args.retrieval_batch_size,
        device=args.device,
    )
    metrics = compute_metrics(qrels, run, args.top_k)
    write_run(args.run_output, run, qrels, query_texts)

    result = {
        "task": "SciRepEval Search full-corpus diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(Path(args.model_path).resolve()),
        "encoder_backend": args.encoder_backend,
        "encoder": (
            flagembedding_model_options(
                args.encoder_backend,
                args.devices if args.devices else args.device,
                args.batch_size,
                args.query_instruction,
                args.query_instruction_format,
            )
            if args.encoder_backend.startswith("flagembedding-")
            else {"prompt_name": "blank"}
        ),
        "metadata": str(Path(args.metadata).resolve()),
        "qrels": str(Path(args.qrels).resolve()),
        "retrieval": {
            "corpus_size": len(corpus_ids),
            "metadata_query_count": len(query_texts),
            "evaluated_query_count": len(query_ids),
            "metadata_candidate_pair_count": metadata_pairs,
            "qrel_pair_count": qrel_pairs,
            "judged_documents_per_query": sorted(judged_counts),
            "top_k": args.top_k,
            "distance": "squared Euclidean (ranking-equivalent to Euclidean)",
            "unjudged_policy": "treated as non-relevant by pytrec_eval",
        },
        "metrics_percent": metrics,
        "run_output": str(Path(args.run_output).resolve()),
    }

    result_path = Path(args.result_output)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
