#!/usr/bin/env python3
"""Convert SciRepEval Search evaluation parquet to FlagEmbedding custom JSONL."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


OUTPUT_NAMES = (
    "corpus.jsonl",
    "test_queries.jsonl",
    "test_qrels.jsonl",
    "conversion_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="SciRepEval Search evaluation parquet file, or a directory containing one.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for FlagEmbedding custom corpus/query/qrel JSONL files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only converter-owned output files if they already exist.",
    )
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Input parquet path does not exist: {path}")
    matches = sorted(path.rglob("evaluation-*.parquet"))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one evaluation-*.parquet below input directory; "
            f"found {len(matches)} under {path}"
        )
    return matches[0]


def required_text(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or empty {field} at {context}")
    return value.strip()


def as_score(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean score at {context}")
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid score at {context}: {value!r}") from exc
    if score < 0:
        raise ValueError(f"Negative score at {context}: {score}")
    return score


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path is not a directory: {path}")
    existing = [path / name for name in OUTPUT_NAMES if (path / name).exists()]
    if existing and not overwrite:
        names = ", ".join(str(item) for item in existing)
        raise FileExistsError(f"Refusing to overwrite existing converter output: {names}")
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for item in existing:
            item.unlink()


def main() -> None:
    args = parse_args()
    input_path = resolve_input(args.input)
    prepare_output(args.output, args.overwrite)

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required; use the embedding evaluation environment.") from exc

    corpus: dict[str, dict[str, str]] = {}
    queries: dict[str, str] = {}
    qrels: dict[tuple[str, str], int] = {}
    total_candidates = 0
    positive_candidates = 0

    parquet_file = pq.ParquetFile(input_path)
    required_columns = {"query", "doc_id", "candidates"}
    missing = required_columns.difference(parquet_file.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing required parquet columns: {sorted(missing)}")

    for batch_index, batch in enumerate(
        parquet_file.iter_batches(columns=["query", "doc_id", "candidates"]), start=1
    ):
        for row_index, row in enumerate(batch.to_pylist(), start=1):
            context = f"batch {batch_index}, row {row_index}"
            query_id = required_text(row.get("doc_id"), "doc_id", context)
            query_text = required_text(row.get("query"), "query", context)
            previous_query = queries.setdefault(query_id, query_text)
            if previous_query != query_text:
                raise ValueError(f"Conflicting query text for doc_id={query_id!r} at {context}")

            candidates = row.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError(f"candidates is not a list at {context}")
            for candidate_index, candidate in enumerate(candidates, start=1):
                candidate_context = f"{context}, candidate {candidate_index}"
                if not isinstance(candidate, dict):
                    raise ValueError(f"Candidate is not an object at {candidate_context}")
                total_candidates += 1
                doc_id = required_text(candidate.get("doc_id"), "candidate.doc_id", candidate_context)
                title = str(candidate.get("title") or "").strip()
                abstract = str(candidate.get("abstract") or "").strip()
                text = "\n\n".join(part for part in (title, abstract) if part)
                if not text:
                    raise ValueError(f"Candidate has neither title nor abstract at {candidate_context}")
                document = {"id": doc_id, "title": title, "text": text}
                previous_document = corpus.setdefault(doc_id, document)
                if previous_document != document:
                    raise ValueError(f"Conflicting content for candidate.doc_id={doc_id!r}")

                score = as_score(candidate.get("score"), candidate_context)
                if score > 0:
                    positive_candidates += 1
                    qrel_key = (query_id, doc_id)
                    qrels[qrel_key] = max(score, qrels.get(qrel_key, 0))

    if not corpus:
        raise ValueError("No corpus documents were produced")
    if not queries:
        raise ValueError("No queries were produced")
    if not qrels:
        raise ValueError("No positive qrels (score > 0) were produced")

    corpus_rows = [corpus[doc_id] for doc_id in sorted(corpus)]
    query_rows = [{"id": query_id, "text": queries[query_id]} for query_id in sorted(queries)]
    qrel_rows = [
        {"qid": query_id, "docid": doc_id, "relevance": score}
        for (query_id, doc_id), score in sorted(qrels.items())
    ]
    write_jsonl(args.output / "corpus.jsonl", corpus_rows)
    write_jsonl(args.output / "test_queries.jsonl", query_rows)
    write_jsonl(args.output / "test_qrels.jsonl", qrel_rows)
    summary = {
        "source": str(input_path.resolve()),
        "corpus": len(corpus_rows),
        "queries": len(query_rows),
        "qrels": len(qrel_rows),
        "total_candidates": total_candidates,
        "positive_candidates": positive_candidates,
        "deduplicated_qrels": positive_candidates - len(qrel_rows),
    }
    (args.output / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
