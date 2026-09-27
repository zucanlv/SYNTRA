#!/usr/bin/env python3
"""Build qpos JSON from CoIR cosqa official train positives.

The output matches the pipeline's qpos / DiverseQuery input format:

{
  "input_path": ".../default/train/data-00000-of-00001.arrow",
  "num_queries": 1,
  "results": [
    {
      "doc_id": "d...",
      "doc": "positive code snippet",
      "queries": [{"query": "natural language programming query"}]
    }
  ]
}

Only train rows with ``score == 1`` are kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.ipc as ipc


DEFAULT_DATASET_DIR = Path(
    "/data/share/project/shared_datasets/DSA/datasets/CoIR-Retrieval__cosqa"
)
DEFAULT_OUTPUT = Path(
    "/data/share/project/shared_datasets/DSA/others/qpos/cosqa/"
    "cosqa-official-train-score1-qpos.json"
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
    out: dict[str, str] = {}
    for row in read_arrow_rows(arrow_path):
        row_id = row.get("_id")
        text = row.get("text")
        if isinstance(row_id, str) and isinstance(text, str):
            out[row_id] = text
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoIR cosqa official train score=1 rows -> qpos JSON."
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
        help=f"Output qpos JSON path. Default: {DEFAULT_OUTPUT}",
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

    results: list[dict] = []
    skipped_missing = 0
    skipped_non_positive = 0
    seen_pairs: set[tuple[str, str]] = set()

    for row in train_rows:
        if row.get("score") != 1:
            skipped_non_positive += 1
            continue
        qid = row.get("query-id")
        cid = row.get("corpus-id")
        q = query_text.get(qid) if isinstance(qid, str) else None
        doc = corpus_text.get(cid) if isinstance(cid, str) else None
        if not q or not doc or not isinstance(cid, str):
            skipped_missing += 1
            continue
        pair = (q, cid)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        results.append(
            {
                "doc_id": cid,
                "doc": doc,
                "queries": [{"query": q}],
            }
        )

    if not results:
        print("No score=1 qpos rows found", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_path": str(train_path.resolve()),
        "num_queries": 1,
        "results": results,
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"train rows read        : {len(train_rows):,}")
    print(f"qpos results written   : {len(results):,}")
    print(f"score!=1 rows skipped  : {skipped_non_positive:,}")
    print(f"missing text skipped   : {skipped_missing:,}")
    print(f"output                 : {args.output}")


if __name__ == "__main__":
    main()
