#!/usr/bin/env python3
"""
Build MSMARCO-style JSONL (prompt, query, pos, neg) from Annotated_Main JSON.

For each ``results[]`` item, take the first query in ``queries[]``, map
``positive_doc_ids``, ``hard_negative_doc_ids``, and ``easy_negative_doc_ids``
to passage text via ``candidates`` (doc_id -> text).

Negative selection rules (after resolving doc_ids to text in ``candidates``):

- Drop if ``pos`` is empty.
- Drop if hard + easy negatives are both empty.
- If hard negatives count is at least ``--target-negatives``, ``neg`` is all
  hard negatives.
- If hard negatives count is below ``--target-negatives``, append easy
  negatives in their original order. If hard + easy negatives still cannot
  reach ``--target-negatives``, keep the record with all resolved easy
  negatives appended.

Usage
-----
Default target negatives is 7:

    python anno_queries_to_msmarco_syn_fill_easy.py \
        --input /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/msmarco_20260330_024548/Annotated_Main_20260330_035435.json \
        --output /data/share/project/shared_datasets/DSA/syn_data/msmarco/msmarco_syn_0330_anno_1_fill_easy.jsonl \
        --prompt "Given a web search query, retrieve relevant passages that answer the query."

Override target negatives:

    python anno_queries_to_msmarco_syn_fill_easy.py \
        --input /path/to/Annotated_Main.json \
        --output /path/to/msmarco_syn_fill_easy.jsonl \
        --prompt "Given a web search query, retrieve relevant passages that answer the query." \
        --target-negatives 5
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
    / "shared_datasets/DSA/syn_data/msmarco/msmarco_syn_0330_anno_1_fill_easy.jsonl"
)
DEFAULT_TARGET_NEGATIVES = 7


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


def _parse_result_pos_hard_easy(
    result: dict,
) -> tuple[str, list[str], list[str], list[str]] | None:
    """Parse query, pos, hard_neg, easy_neg from one result item."""
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


_SKIP_PARSE = "skip_parse"
_SKIP_POS_EMPTY = "skip_pos_empty"
_SKIP_NEG_EMPTY = "skip_neg_empty"
_OK_NEG_HARD_ONLY = "ok_neg_hard_only"
_OK_NEG_HARD_EASY_FILL = "ok_neg_hard_easy_fill"
_OK_NEG_HARD_EASY_PARTIAL = "ok_neg_hard_easy_partial"


def _select_negatives(
    hard_neg: list[str],
    easy_neg: list[str],
    target_negatives: int,
) -> tuple[list[str] | None, str]:
    if not hard_neg and not easy_neg:
        return None, _SKIP_NEG_EMPTY
    if len(hard_neg) >= target_negatives:
        return hard_neg, _OK_NEG_HARD_ONLY

    needed = target_negatives - len(hard_neg)
    filled_neg = hard_neg + easy_neg[:needed]
    if len(filled_neg) < target_negatives:
        return filled_neg, _OK_NEG_HARD_EASY_PARTIAL
    return filled_neg, _OK_NEG_HARD_EASY_FILL


def _build_record(
    result: dict,
    prompt: str,
    target_negatives: int,
) -> tuple[dict | None, str]:
    parsed = _parse_result_pos_hard_easy(result)
    if parsed is None:
        return None, _SKIP_PARSE
    q, pos, hard_neg, easy_neg = parsed
    if not pos:
        return None, _SKIP_POS_EMPTY

    neg, outcome = _select_negatives(hard_neg, easy_neg, target_negatives)
    if neg is None:
        return None, outcome
    return (
        {
            "prompt": prompt,
            "query": q,
            "pos": pos,
            "neg": neg,
        },
        outcome,
    )


def result_to_record(
    result: dict,
    prompt: str,
    target_negatives: int = DEFAULT_TARGET_NEGATIVES,
) -> dict | None:
    rec, _ = _build_record(result, prompt, target_negatives)
    return rec


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Annotated_Main JSON -> MSMARCO JSONL with easy-negative fill"
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
    p.add_argument(
        "--target-negatives",
        type=_positive_int,
        default=DEFAULT_TARGET_NEGATIVES,
        help="minimum negative passages per record; hard negatives are filled with easy negatives when needed",
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
    n_easy_fill = 0
    n_easy_partial = 0

    with args.output.open("w", encoding="utf-8") as out:
        for item in results:
            if not isinstance(item, dict):
                n_skip += 1
                continue
            rec, outcome = _build_record(item, args.prompt, args.target_negatives)
            if rec is None:
                if outcome == _SKIP_POS_EMPTY:
                    n_pos_empty += 1
                elif outcome == _SKIP_NEG_EMPTY:
                    n_neg_empty += 1
                n_skip += 1
                continue
            if outcome == _OK_NEG_HARD_EASY_FILL:
                n_easy_fill += 1
            elif outcome == _OK_NEG_HARD_EASY_PARTIAL:
                n_easy_partial += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    logger.info(
        "done: results=%s written=%s skipped=%s pos_empty=%s "
        "neg_empty=%s easy_fill=%s easy_partial=%s target_negatives=%s -> %s",
        n_in,
        n_out,
        n_skip,
        n_pos_empty,
        n_neg_empty,
        n_easy_fill,
        n_easy_partial,
        args.target_negatives,
        args.output,
    )


if __name__ == "__main__":
    main()
