#!/usr/bin/env python3
"""
Build MSMARCO-style JSONL (prompt, query, pos, neg) from Annotated_Main JSON.

For each ``results[]`` item, take the first query in ``queries[]``, map
``positive_doc_ids`` and ``hard_negative_doc_ids`` to passage text via
``candidates`` (doc_id -> text). ``neg`` is hard negatives only.

Records are skipped when ``pos`` or ``neg`` would be empty (after resolving
doc_ids to text present in ``candidates``).

Usage
-----
    python anno_queries_to_msmarco_syn.py \
        --input /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/msmarco_20260330_024548/Annotated_Main_20260330_035435.json \
        --output ./test.jsonl --prompt "Given a web search query, retrieve relevant passages that answer the query."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Historical defaults; explicit --input/--output work at any checkout depth.
_PROJECT_ROOT = Path("/data/share/project")
DEFAULT_INPUT = (
    _PROJECT_ROOT
    / "zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results"
    "/msmarco_20260330_024142/Annotated_Main_20260330_024215.json"
)
DEFAULT_OUTPUT = (
    _PROJECT_ROOT
    / "shared_datasets/DSA/syn_data/msmarco/msmarco_syn_0330_anno_1.jsonl"
)

def _candidate_map(candidates: object) -> dict[str, str]:
    if not isinstance(candidates, list):
        return {}
    out: dict[str, str] = {}
    for c in candidates:
        if not isinstance(c, dict):
            continue
        did = c.get("doc_id")
        doc = c.get("doc")
        if isinstance(did, str) and isinstance(doc, str) and did:
            out[did] = doc
    return out


def _texts_for_ids(doc_ids: object, id_to_text: dict[str, str]) -> list[str]:
    if not isinstance(doc_ids, list):
        return []
    texts: list[str] = []
    for did in doc_ids:
        if not isinstance(did, str):
            continue
        t = id_to_text.get(did)
        if t is not None:
            texts.append(t)
    return texts


def _parse_result_pos_neg(result: dict) -> tuple[str, list[str], list[str]] | None:
    """Parse query, pos, neg from one result item. Returns None if unusable."""
    queries = result.get("queries")
    if not isinstance(queries, list) or not queries:
        return None
    qobj = queries[0]
    if not isinstance(qobj, dict):
        return None
    q = qobj.get("query")
    if not isinstance(q, str) or not q.strip():
        return None

    id_to_text = _candidate_map(qobj.get("candidates"))
    pos = _texts_for_ids(qobj.get("positive_doc_ids"), id_to_text)
    neg = _texts_for_ids(qobj.get("hard_negative_doc_ids"), id_to_text)
    return (q.strip(), pos, neg)


def result_to_record(result: dict, prompt: str) -> dict | None:
    parsed = _parse_result_pos_neg(result)
    if parsed is None:
        return None
    q, pos, neg = parsed
    if not pos or not neg:
        return None
    return {
        "prompt": prompt,
        "query": q,
        "pos": pos,
        "neg": neg,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Annotated_Main JSON -> msmarco_syn_0330_anno_1.jsonl"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Annotated JSON")
    p.add_argument(
        "--prompt",
        required=True,
        help="MSMARCO-style instruction prompt (required)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output JSONL path",
    )
    args = p.parse_args()

    if not args.input.is_file():
        logger.error("input not found: %s", args.input)
        sys.exit(1)

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results")
    if not isinstance(results, list):
        logger.error("missing or invalid 'results' array")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_in = len(results)
    n_out = 0
    n_skip = 0
    n_pos_empty = 0
    n_neg_empty = 0

    with args.output.open("w", encoding="utf-8") as out:
        for item in results:
            if not isinstance(item, dict):
                n_skip += 1
                continue
            parsed = _parse_result_pos_neg(item)
            if parsed is None:
                n_skip += 1
                continue
            _q, pos, neg = parsed
            if not pos:
                n_pos_empty += 1
            if not neg:
                n_neg_empty += 1
            if not pos or not neg:
                n_skip += 1
                continue
            rec = {
                "prompt": args.prompt,
                "query": _q,
                "pos": pos,
                "neg": neg,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    logger.info(
        "done: results=%s written=%s skipped=%s pos_empty=%s neg_empty=%s -> %s",
        n_in,
        n_out,
        n_skip,
        n_pos_empty,
        n_neg_empty,
        args.output,
    )


if __name__ == "__main__":
    main()
