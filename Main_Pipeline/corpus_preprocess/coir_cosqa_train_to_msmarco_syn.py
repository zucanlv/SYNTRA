#!/usr/bin/env python3
"""Convert CoIR cosqa official train split to prompt/query/pos/neg JSONL.

Rows are grouped by ``query-id``. For each query, train rows with ``score == 1``
are written to ``pos`` and rows with ``score == 0`` are written to ``neg``.

Important: the local CoIR cosqa train split currently has each ``query-id`` only
once. That means the official split contains single-sided query/doc labels:
some queries have only ``pos`` and some have only ``neg``. By default this
script writes only complete training records where both lists are non-empty,
which may produce zero rows for this dataset. Use ``--keep-incomplete`` only if
you explicitly want rows with empty ``pos`` or empty ``neg``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.ipc as ipc


DEFAULT_DATASET_DIR = Path(
    "/data/share/project/shared_datasets/DSA/datasets/CoIR-Retrieval__cosqa"
)
DEFAULT_OUTPUT = Path(
    "/data/share/project/shared_datasets/DSA/syn_data/"
    "RQ2-code-retrieve-task/cosqa-official-train-by-query.jsonl"
)
DEFAULT_PROMPT = (
    "Given a natural language programming query, retrieve relevant code "
    "snippets that correctly implement, solve, or answer the query."
)


def read_arrow_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Arrow file not found: {path}")
    rows: list[dict] = []
    with path.open("rb") as f:
        reader = ipc.open_stream(f)
        for batch in reader:
            rows.extend(batch.to_pylist())
    return rows


def load_text_index(arrow_path: Path) -> dict[str, str]:
    rows = read_arrow_rows(arrow_path)
    out: dict[str, str] = {}
    for row in rows:
        row_id = row.get("_id")
        text = row.get("text")
        if isinstance(row_id, str) and isinstance(text, str):
            out[row_id] = text
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoIR cosqa official train split -> MSMARCO-style JSONL."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Local CoIR-Retrieval__cosqa directory. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Instruction prompt written into every JSONL record.",
    )
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Also write rows whose pos or neg list is empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    train_path = dataset_dir / "default/train/data-00000-of-00001.arrow"
    queries_path = dataset_dir / "queries/queries/data-00000-of-00001.arrow"
    corpus_path = dataset_dir / "corpus/corpus/data-00000-of-00001.arrow"

    train_rows = read_arrow_rows(train_path)
    query_text = load_text_index(queries_path)
    corpus_text = load_text_index(corpus_path)

    grouped: dict[str, dict[str, object]] = {}
    skipped_missing = 0
    score_counts: dict[int, int] = defaultdict(int)

    for row in train_rows:
        qid = row.get("query-id")
        cid = row.get("corpus-id")
        score = row.get("score")
        q = query_text.get(qid) if isinstance(qid, str) else None
        doc = corpus_text.get(cid) if isinstance(cid, str) else None
        if not q or not doc:
            skipped_missing += 1
            continue
        if score not in (0, 1):
            raise ValueError(f"Unexpected score {score!r} in row: {row}")

        score_counts[score] += 1
        rec = grouped.setdefault(qid, {"query": q, "pos": [], "neg": []})
        field = "pos" if score == 1 else "neg"
        rec[field].append(doc)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    incomplete_pos = 0
    incomplete_neg = 0

    with args.output.open("w", encoding="utf-8") as out:
        for rec in grouped.values():
            pos = rec["pos"]
            neg = rec["neg"]
            if not pos:
                incomplete_pos += 1
            if not neg:
                incomplete_neg += 1
            if not args.keep_incomplete and (not pos or not neg):
                continue

            out.write(
                json.dumps(
                    {
                        "prompt": args.prompt,
                        "query": rec["query"],
                        "pos": pos,
                        "neg": neg,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    print(f"train rows read          : {len(train_rows):,}")
    print(f"unique queries grouped   : {len(grouped):,}")
    print(f"score=1 rows             : {score_counts[1]:,}")
    print(f"score=0 rows             : {score_counts[0]:,}")
    print(f"queries missing pos      : {incomplete_pos:,}")
    print(f"queries missing neg      : {incomplete_neg:,}")
    print(f"rows written             : {n_written:,}")
    print(f"skipped missing text     : {skipped_missing:,}")
    print(f"output                   : {args.output}")
    if n_written == 0 and not args.keep_incomplete:
        print(
            "No complete rows were written because no query has both pos and neg "
            "inside the official train split.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
