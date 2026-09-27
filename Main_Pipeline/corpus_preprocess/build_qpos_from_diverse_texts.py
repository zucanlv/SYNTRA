#!/usr/bin/env python3
"""Build qpos JSON from diverse_texts JSONL and MSMARCO hn train JSONL.

The output matches the current qpos format used by the main pipeline:

{
  "input_path": ".../diverse_texts.jsonl",
  "num_queries": 1,
  "results": [
    {
      "doc_id": "msmarco_...",
      "doc": "...",
      "queries": [{"query": "..."}]
    }
  ]
}

Default inputs are the current DSA diverse_texts sample80k and MSMARCO training set.
By default, the first matching training query is kept for each doc, matching the
existing qpos files where num_queries is 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from production_sample_utils import apply_production_sample  # noqa: E402


DEFAULT_DIVERSE_TEXTS = Path(
    "/data/share/project/shared_datasets/DSA/others/diverse_texts/"
    "diverse_texts_cos062_training_corpus_499996_93804_sample80k.jsonl"
)
DEFAULT_TRAINING_SET = Path(
    "/data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/MSMARCO/"
    "msmarco_hn_train.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/data/share/project/shared_datasets/DSA/others/qpos/"
    "diverse_texts_cos062_training_corpus_499996_93804_sample80k_qpos_new.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diverse-texts",
        type=Path,
        default=DEFAULT_DIVERSE_TEXTS,
        help=f"Diverse texts JSONL. Default: {DEFAULT_DIVERSE_TEXTS}",
    )
    parser.add_argument(
        "--training-set",
        type=Path,
        default=DEFAULT_TRAINING_SET,
        help=f"MSMARCO hn train JSONL. Default: {DEFAULT_TRAINING_SET}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output qpos JSON. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Optional production_sample random count. Omit to use all diverse_texts rows.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --mode random.")
    parser.add_argument(
        "--mode",
        choices=["random", "range", "indices"],
        default="random",
        help="Sampling mode shared with main pipeline production_sample.",
    )
    parser.add_argument("--range", type=str, default=None, help="mode=range: 'start-end' closed interval.")
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="mode=indices: comma-separated 0-based row indices.",
    )
    parser.add_argument(
        "--max-queries-per-doc",
        type=int,
        default=1,
        help="Keep up to N matched queries per doc. Use 0 to keep all matches.",
    )
    parser.add_argument(
        "--split-parts",
        type=int,
        default=None,
        metavar="N",
        help="Optionally split results into N contiguous qpos files: output_part01.json ...",
    )
    parser.add_argument(
        "--write-combined",
        action="store_true",
        help="With --split-parts, also write the combined qpos JSON to --output.",
    )
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=None,
        help="Optional JSONL path for diverse_texts rows that did not match any training pos.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def get_doc_text(row: dict) -> str:
    """Return the passage text from current or legacy diverse_texts schemas."""
    doc = row.get("doc")
    if isinstance(doc, str):
        return doc
    text = row.get("text")
    if isinstance(text, str):
        title = row.get("title")
        if isinstance(title, str) and title:
            return f"{title} {text}".strip()
        return text
    return ""


def split_evenly(items: list[dict], n_parts: int) -> list[list[dict]]:
    if n_parts < 1:
        raise ValueError("--split-parts must be >= 1")
    total = len(items)
    base = total // n_parts
    rem = total % n_parts
    chunks: list[list[dict]] = []
    start = 0
    for i in range(n_parts):
        size = base + (1 if i < rem else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks


def match_queries_from_training_set(
    training_set: Path,
    needed_texts: set[str],
    max_queries_per_doc: int,
    logger: logging.Logger,
) -> dict[str, list[str]]:
    """Stream training_set and build doc text -> matched query list."""
    text_to_queries: dict[str, list[str]] = defaultdict(list)
    unlimited = max_queries_per_doc == 0
    n_lines = 0
    n_pos_seen = 0

    with training_set.open("r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query = obj.get("query")
            if not isinstance(query, str) or not query.strip():
                continue
            pos_list = obj.get("pos") or []
            if not isinstance(pos_list, list):
                continue
            for pos in pos_list:
                if not isinstance(pos, str):
                    continue
                n_pos_seen += 1
                if pos not in needed_texts:
                    continue
                queries = text_to_queries[pos]
                if query in queries:
                    continue
                if unlimited or len(queries) < max_queries_per_doc:
                    queries.append(query)

            if not unlimited and len(text_to_queries) >= len(needed_texts):
                if all(len(qs) >= max_queries_per_doc for qs in text_to_queries.values()):
                    break

            if n_lines % 500_000 == 0:
                logger.info(
                    "Scanned %d training rows; matched %d / %d distinct docs",
                    n_lines,
                    len(text_to_queries),
                    len(needed_texts),
                )

    logger.info(
        "Finished scanning %d training rows (%d pos entries); matched %d / %d distinct docs",
        n_lines,
        n_pos_seen,
        len(text_to_queries),
        len(needed_texts),
    )
    return dict(text_to_queries)


def write_qpos(path: Path, input_path: Path, num_queries: int, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "input_path": str(input_path.resolve()),
        "num_queries": num_queries,
        "results": results,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_missing(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("build_qpos_from_diverse_texts")

    if not args.diverse_texts.is_file():
        logger.error("diverse-texts not found: %s", args.diverse_texts)
        sys.exit(1)
    if not args.training_set.is_file():
        logger.error("training-set not found: %s", args.training_set)
        sys.exit(1)
    if args.max_queries_per_doc < 0:
        logger.error("--max-queries-per-doc must be >= 0")
        sys.exit(1)

    docs = load_jsonl(args.diverse_texts)
    logger.info("Loaded %d diverse_texts rows from %s", len(docs), args.diverse_texts)

    sample_cfg: dict = {"mode": args.mode, "count": args.count, "seed": args.seed}
    if args.range is not None:
        sample_cfg["range"] = args.range
    if args.indices is not None:
        sample_cfg["indices"] = args.indices
    sampled = apply_production_sample(docs, sample_cfg, logger)
    if not sampled:
        logger.error("Sampling returned no rows")
        sys.exit(1)

    needed_texts = {get_doc_text(row) for row in sampled if get_doc_text(row)}
    logger.info("Need to match %d distinct doc texts from %d sampled rows", len(needed_texts), len(sampled))

    text_to_queries = match_queries_from_training_set(
        args.training_set,
        needed_texts,
        args.max_queries_per_doc,
        logger,
    )

    results: list[dict] = []
    missing_rows: list[dict] = []
    for row in sampled:
        doc = get_doc_text(row)
        queries = text_to_queries.get(doc, [])
        if not doc or not queries:
            missing_rows.append(row)
            continue
        results.append(
            {
                "doc_id": row.get("doc_id", ""),
                "doc": doc,
                "queries": [{"query": q} for q in queries],
            }
        )

    if missing_rows:
        logger.warning("Missing matches for %d / %d sampled rows", len(missing_rows), len(sampled))
        if args.missing_output:
            write_missing(args.missing_output, missing_rows)
            logger.info("Wrote missing rows to %s", args.missing_output)

    if not results:
        logger.error("No matched qpos results; refusing to write output")
        sys.exit(1)

    if args.max_queries_per_doc == 0:
        num_queries = max(len(item["queries"]) for item in results)
    else:
        num_queries = args.max_queries_per_doc

    if args.split_parts is not None:
        chunks = split_evenly(results, args.split_parts)
        suffix = args.output.suffix or ".json"
        for idx, chunk in enumerate(chunks, start=1):
            part_path = args.output.parent / f"{args.output.stem}_part{idx:02d}{suffix}"
            write_qpos(part_path, args.diverse_texts, num_queries, chunk)
            logger.info("Wrote %s (%d results)", part_path, len(chunk))
        if args.write_combined:
            write_qpos(args.output, args.diverse_texts, num_queries, results)
            logger.info("Wrote combined %s (%d results)", args.output, len(results))
    else:
        write_qpos(args.output, args.diverse_texts, num_queries, results)
        logger.info("Wrote %s (%d results)", args.output, len(results))


if __name__ == "__main__":
    main()
