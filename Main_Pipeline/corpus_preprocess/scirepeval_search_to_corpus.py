#!/usr/bin/env python3
"""Build a train-only scientific-paper corpus from SciRepEval Search.

The Search subset stores one user query and a nested candidate-paper list per
row. This converter streams only the official ``train`` split, deduplicates
candidate papers by ``doc_id``, preserves the official document fields, and
writes the evaluator-aligned document text consumed by the local pipeline.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


DEFAULT_OUTPUT = Path(
    "/data/share/project/shared_datasets/DSA/datasets/scirepeval-search/"
    "search-train-corpus.jsonl"
)

SEARCH_DOCUMENT_FIELDS = ("title", "abstract", "venue", "year")
# The configured embedding model is BGE-M3, whose EOS/SEP token is </s>.
SEARCH_DOCUMENT_SEPARATOR = " </s> "


@contextmanager
def atomic_text_writer(path: Path) -> Iterator[TextIO]:
    """Write beside *path* and replace it only after successful completion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            yield output
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def render_search_document(candidate: dict[str, Any]) -> str:
    """Mirror SciRepEval's ordered, non-empty Search field concatenation."""
    parts = [
        str(candidate[field])
        for field in SEARCH_DOCUMENT_FIELDS
        if candidate.get(field)
    ]
    return SEARCH_DOCUMENT_SEPARATOR.join(parts).strip()


def _corpus_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": candidate["doc_id"],
        "title": candidate["title"],
        "abstract": candidate["abstract"],
        "text": render_search_document(candidate),
        "corpus_id": candidate.get("corpus_id"),
        "venue": candidate.get("venue", ""),
        "year": candidate.get("year"),
        "author_names": candidate.get("author_names", []),
        "n_citations": candidate.get("n_citations"),
        "n_key_citations": candidate.get("n_key_citations"),
        "split": "train",
    }


def write_search_corpus(
    rows: Iterable[dict[str, Any]],
    output_path: Path,
    *,
    progress_every: int = 10_000,
) -> dict[str, int]:
    """Flatten all Search candidates and deduplicate only by document ID."""
    stats = {
        "search_rows": 0,
        "candidates": 0,
        "documents_written": 0,
        "duplicate_doc_ids_skipped": 0,
    }
    seen_doc_ids: set[str] = set()

    with atomic_text_writer(output_path) as output:
        for row in rows:
            stats["search_rows"] += 1
            candidates = row.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError(
                    f"Search row {stats['search_rows']} has no candidate list"
                )

            for candidate in candidates:
                stats["candidates"] += 1
                if not isinstance(candidate, dict):
                    raise ValueError(
                        f"Search row {stats['search_rows']} contains a non-object candidate"
                    )
                doc_id = candidate.get("doc_id")
                title = candidate.get("title")
                abstract = candidate.get("abstract")
                if (
                    doc_id is None
                    or not str(doc_id).strip()
                    or not isinstance(title, str)
                    or not isinstance(abstract, str)
                ):
                    raise ValueError(
                        f"Search row {stats['search_rows']} contains a candidate with "
                        "an invalid doc_id, title, or abstract"
                    )

                normalised_doc_id = str(doc_id)
                if normalised_doc_id in seen_doc_ids:
                    stats["duplicate_doc_ids_skipped"] += 1
                    continue
                seen_doc_ids.add(normalised_doc_id)

                corpus_row = _corpus_row(candidate)
                output.write(
                    json.dumps(corpus_row, ensure_ascii=False, separators=(",", ":"))
                )
                output.write("\n")
                stats["documents_written"] += 1

            if progress_every and stats["search_rows"] % progress_every == 0:
                print(json.dumps(stats, sort_keys=True), flush=True)

    return stats


def iter_official_train_rows(
    dataset_name: str = "allenai/scirepeval",
    config_name: str = "search",
) -> Iterable[dict[str, Any]]:
    """Stream the official Search train split without loading evaluation data."""
    from datasets import load_dataset

    return load_dataset(
        dataset_name,
        config_name,
        split="train",
        streaming=True,
        trust_remote_code=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="allenai/scirepeval")
    parser.add_argument("--config", default="search")
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = iter_official_train_rows(args.dataset, args.config)
    stats = write_search_corpus(
        rows,
        args.output,
        progress_every=args.progress_every,
    )
    print(json.dumps({"output": str(args.output), **stats}, indent=2))


if __name__ == "__main__":
    main()
