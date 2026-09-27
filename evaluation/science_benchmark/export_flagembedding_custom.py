#!/usr/bin/env python3
"""Export BEIR-style converted tasks to FlagEmbedding's native custom layout."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


TASKS = (
    "birco",
    "birco_arguana",
    "birco_clinical_trial",
    "birco_doris_mae",
    "birco_relic",
    "birco_whatsthatbook",
    "evidencebench",
    "financebench_open_source_subset",
    "kilt_retrieval___provenance_tasks",
    "kuake_qtr",
    "lecard",
    "legalbench_rag",
    "legalbench_rag_cuad",
    "legalcitebench_citation_retrieval",
    "trialgpt_retrieval_benchmark",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{line_no}")
                yield row


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_id(row: dict, path: Path, line_no: int) -> str:
    value = row.get("id", row.get("_id"))
    if value in (None, ""):
        raise ValueError(f"Missing id/_id at {path}:{line_no}")
    return str(value)


def export_task(source_dir: Path, output_dir: Path, overwrite: bool) -> dict[str, int]:
    corpus_source = source_dir / "corpus.jsonl"
    queries_source = source_dir / "queries.jsonl"
    qrels_source = source_dir / "qrels" / "test.tsv"
    required = (corpus_source, queries_source, qrels_source)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required source files: " + ", ".join(missing))

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    corpus_rows = []
    corpus_ids: set[str] = set()
    for line_no, row in enumerate(read_jsonl(corpus_source), 1):
        doc_id = source_id(row, corpus_source, line_no)
        if doc_id in corpus_ids:
            raise ValueError(f"Duplicate corpus id {doc_id!r} in {corpus_source}")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing text for corpus id {doc_id!r}")
        corpus_ids.add(doc_id)
        corpus_rows.append({"id": doc_id, "title": str(row.get("title") or ""), "text": text})

    query_rows = []
    empty_queries = 0
    query_ids: set[str] = set()
    for line_no, row in enumerate(read_jsonl(queries_source), 1):
        query_id = source_id(row, queries_source, line_no)
        if query_id in query_ids:
            raise ValueError(f"Duplicate query id {query_id!r} in {queries_source}")
        text = row.get("text", row.get("query"))
        if not isinstance(text, str):
            raise ValueError(f"Missing text for query id {query_id!r}")
        if not text.strip():
            empty_queries += 1
        query_ids.add(query_id)
        query_rows.append({"id": query_id, "text": text})

    qrel_rows = []
    dangling_qrel_queries = 0
    dangling_qrel_docs = 0
    with qrels_source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"query-id", "corpus-id", "score"}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError(f"Expected TSV columns {sorted(expected)} in {qrels_source}")
        for line_no, row in enumerate(reader, 2):
            query_id = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            if query_id not in query_ids:
                dangling_qrel_queries += 1
            if doc_id not in corpus_ids:
                dangling_qrel_docs += 1
            try:
                relevance = int(float(row["score"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid qrels score at {qrels_source}:{line_no}") from exc
            qrel_rows.append({"qid": query_id, "docid": doc_id, "relevance": relevance})
    if not qrel_rows:
        raise ValueError(f"No qrels in {qrels_source}")

    write_jsonl(output_dir / "corpus.jsonl", corpus_rows)
    write_jsonl(output_dir / "test_queries.jsonl", query_rows)
    write_jsonl(output_dir / "test_qrels.jsonl", qrel_rows)
    return {
        "corpus": len(corpus_rows),
        "queries": len(query_rows),
        "qrels": len(qrel_rows),
        "empty_queries": empty_queries,
        "dangling_qrel_queries": dangling_qrel_queries,
        "dangling_qrel_docs": dangling_qrel_docs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("science_benchmark/converted_full"))
    parser.add_argument("--output-root", type=Path, default=Path("science_benchmark/converted_flagembedding_custom"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = {}
    for task in args.tasks:
        summary[task] = export_task(args.source_root / task, args.output_root / task, args.overwrite)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "export_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
