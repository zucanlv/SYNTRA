#!/usr/bin/env python3
"""
Split HnMine JSONL into eight MSMARCO-style JSONL files (prompt, query, pos, neg).

Each input line is expected to contain a ``queries`` list; entry ``queries[i]``
(0-based) is written to ``msmarco_syn_0328_{i+1}.jsonl`` when present. Records
with fewer than eight queries only populate the earlier files.

``prompt`` is fixed to the MSMARCO instruction string used in
``msmarco_hn_train_sample10k.jsonl``.

Usage
-----
    python hnmine_queries_to_msmarco_syn.py \
        --input /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/msmarco_20260329_055026/HnMine_Main_20260329_120426.jsonl \
        --output-dir /data/share/project/shared_datasets/DSA/syn_data/msmarco
        --prompt "Given a web search query, retrieve relevant passages that answer the query."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT = (
    "/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/msmarco_20260330_191414/HnMine_Main_20260330_191509.jsonl"
)
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "shared_datasets/DSA/syn_data/msmarco"

NUM_SPLITS = 1
OUT_TEMPLATE = "msmarco_syn_{}.jsonl"


def _as_str_list(x: object, field: str) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        out: list[str] = []
        for i, item in enumerate(x):
            if not isinstance(item, str):
                raise TypeError(f"{field}[{i}] is not str")
            out.append(item)
        return out
    raise TypeError(f"{field} must be str or list[str]")


def query_item_to_record(item: dict, prompt: str) -> dict:
    if "query" not in item:
        raise KeyError("query")
    q = item["query"]
    if not isinstance(q, str):
        raise TypeError("query must be str")
    return {
        "prompt": prompt,
        "query": q,
        "pos": _as_str_list(item.get("pos"), "pos"),
        "neg": _as_str_list(item.get("neg"), "neg"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Split HnMine queries[] into msmarco_syn_0328_1..8.jsonl"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="HnMine JSONL")
    p.add_argument(
        "--prompt",
        required=True,
        help="MSMARCO-style instruction prompt (required)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for {OUT_TEMPLATE.format(1)} … {OUT_TEMPLATE.format(NUM_SPLITS)}",
    )
    args = p.parse_args()

    if not args.input.is_file():
        logger.error("input not found: %s", args.input)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [
        args.output_dir / OUT_TEMPLATE.format(i) for i in range(1, NUM_SPLITS + 1)
    ]
    files = [path.open("w", encoding="utf-8") for path in out_paths]
    written = [0] * NUM_SPLITS
    n_bad = n_missing_queries = 0

    try:
        with args.input.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                if not isinstance(obj, dict):
                    n_bad += 1
                    continue
                queries = obj.get("queries")
                if not isinstance(queries, list):
                    n_missing_queries += 1
                    continue
                for slot in range(min(len(queries), NUM_SPLITS)):
                    try:
                        rec = query_item_to_record(queries[slot], args.prompt)
                    except (KeyError, TypeError) as e:
                        logger.warning("line %s slot %s: %s", line_no, slot + 1, e)
                        continue
                    files[slot].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written[slot] += 1
    finally:
        for fh in files:
            fh.close()

    logger.info(
        "done: bad_json=%s rows_without_queries_list=%s per_file_written=%s",
        n_bad,
        n_missing_queries,
        written,
    )
    for i, path in enumerate(out_paths, start=1):
        logger.info("wrote %s (%s lines)", path, written[i - 1])


if __name__ == "__main__":
    main()
