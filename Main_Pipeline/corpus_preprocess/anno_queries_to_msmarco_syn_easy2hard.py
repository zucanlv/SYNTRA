#!/usr/bin/env python3
"""
Build MSMARCO-style JSONL (prompt, query, pos, neg) from Annotated_Main JSON.

For each ``results[]`` item, take the first query in ``queries[]``, map
``positive_doc_ids``, ``hard_negative_doc_ids``, and ``easy_negative_doc_ids``
to passage text via ``candidates`` (doc_id -> text).

Keep / drop rules (after resolving doc_ids to text in ``candidates``):

- Drop if ``pos`` is empty.
- Drop if ``pos`` is non-empty but both hard negatives and easy negatives
  resolve to empty.
- If ``pos`` is non-empty and hard negatives are non-empty, ``neg`` is **only**
  hard negatives (easy negatives are never mixed in).
- If ``pos`` is non-empty, hard negatives are empty, but easy negatives are
  non-empty, ``neg`` is easy negatives.

Usage
-----
    python anno_queries_to_msmarco_syn.py \
        --input /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/msmarco_20260330_024548/Annotated_Main_20260330_035435.json \
        --output /data/share/project/shared_datasets/DSA/syn_data/msmarco/msmarco_syn_0330_anno_1_easy2hard.jsonl --prompt "Given a web search query, retrieve relevant passages that answer the query."
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


def _parse_result_pos_hard_easy(result: dict) -> tuple[str, list[str], list[str], list[str]] | None:
    """Parse query, pos, hard_neg, easy_neg from one result item. Returns None if unusable."""
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
    hard_neg = _texts_for_ids(qobj.get("hard_negative_doc_ids"), id_to_text)
    easy_neg = _texts_for_ids(qobj.get("easy_negative_doc_ids"), id_to_text)
    return (q.strip(), pos, hard_neg, easy_neg)


# Outcome tags for logging; single source of truth for keep/drop + neg assignment.
_SKIP_PARSE = "skip_parse"
_SKIP_POS_EMPTY = "skip_pos_empty"
_SKIP_BOTH_NEGS_EMPTY = "skip_both_negs_empty"
_OK_NEG_HARD = "ok_neg_hard"
_OK_NEG_EASY_FALLBACK = "ok_neg_easy_fallback"


def _build_record(result: dict, prompt: str) -> tuple[dict | None, str]:
    """
    Build one JSONL record or explain why it was skipped.

    Rules (after resolving doc_ids to passage text):
    - Skip if ``pos`` is empty.
    - Skip if ``pos`` is non-empty but both hard and easy negatives are empty.
    - If ``pos`` is non-empty and hard negatives are non-empty: ``neg`` is only hard
      negatives (easy negatives are never mixed in).
    - If ``pos`` is non-empty, hard negatives are empty, easy negatives non-empty:
      ``neg`` is easy negatives.
    """
    parsed = _parse_result_pos_hard_easy(result)
    if parsed is None:
        return None, _SKIP_PARSE
    q, pos, hard_neg, easy_neg = parsed
    if not pos:
        return None, _SKIP_POS_EMPTY
    if hard_neg:
        return (
            {
                "prompt": prompt,
                "query": q,
                "pos": pos,
                "neg": hard_neg,
            },
            _OK_NEG_HARD,
        )
    if not easy_neg:
        return None, _SKIP_BOTH_NEGS_EMPTY
    return (
        {
            "prompt": prompt,
            "query": q,
            "pos": pos,
            "neg": easy_neg,
        },
        _OK_NEG_EASY_FALLBACK,
    )


def result_to_record(result: dict, prompt: str) -> dict | None:
    rec, _ = _build_record(result, prompt)
    return rec


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
    n_hard_and_easy_empty = 0
    n_easy_fallback = 0

    with args.output.open("w", encoding="utf-8") as out:
        for item in results:
            if not isinstance(item, dict):
                n_skip += 1
                continue
            rec, outcome = _build_record(item, args.prompt)
            if rec is None:
                if outcome == _SKIP_POS_EMPTY:
                    n_pos_empty += 1
                elif outcome == _SKIP_BOTH_NEGS_EMPTY:
                    n_hard_and_easy_empty += 1
                n_skip += 1
                continue
            if outcome == _OK_NEG_EASY_FALLBACK:
                n_easy_fallback += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    logger.info(
        "done: results=%s written=%s skipped=%s pos_empty=%s "
        "hard_and_easy_empty=%s easy_fallback=%s -> %s",
        n_in,
        n_out,
        n_skip,
        n_pos_empty,
        n_hard_and_easy_empty,
        n_easy_fallback,
        args.output,
    )


if __name__ == "__main__":
    main()
