#!/usr/bin/env python3
"""Build MSMARCO-style JSONL using only score-3 annotations as positives.

Hard negatives are kept from ``hard_negative_doc_ids`` exactly as in
``anno_queries_to_msmarco_syn.py``. Records without a resolvable score-3
positive or hard negative are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


logger = logging.getLogger(__name__)


def _candidate_map(candidates: object) -> dict[str, str]:
    if not isinstance(candidates, list):
        return {}
    out: dict[str, str] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        doc_id = candidate.get("doc_id")
        doc = candidate.get("doc")
        if isinstance(doc_id, str) and doc_id and isinstance(doc, str):
            out[doc_id] = doc
    return out


def _score3_doc_ids(annotations: object) -> list[str]:
    if not isinstance(annotations, list):
        return []
    doc_ids: list[str] = []
    seen: set[str] = set()
    for annotation in annotations:
        if not isinstance(annotation, dict) or annotation.get("score") != 3:
            continue
        doc_id = annotation.get("doc_id")
        if isinstance(doc_id, str) and doc_id and doc_id not in seen:
            doc_ids.append(doc_id)
            seen.add(doc_id)
    return doc_ids


def _texts_for_ids(doc_ids: object, id_to_text: dict[str, str]) -> list[str]:
    if not isinstance(doc_ids, list):
        return []
    texts: list[str] = []
    for doc_id in doc_ids:
        if not isinstance(doc_id, str):
            continue
        text = id_to_text.get(doc_id)
        if text is not None:
            texts.append(text)
    return texts


def _parse_result_pos_neg(result: dict) -> tuple[str, list[str], list[str]] | None:
    queries = result.get("queries")
    if not isinstance(queries, list) or not queries:
        return None
    query_obj = queries[0]
    if not isinstance(query_obj, dict):
        return None
    query = query_obj.get("query")
    if not isinstance(query, str) or not query.strip():
        return None

    id_to_text = _candidate_map(query_obj.get("candidates"))
    score3_ids = _score3_doc_ids(query_obj.get("annotations"))
    positives = _texts_for_ids(score3_ids, id_to_text)
    negatives = _texts_for_ids(query_obj.get("hard_negative_doc_ids"), id_to_text)
    return query.strip(), positives, negatives


def result_to_record(result: dict, prompt: str) -> dict | None:
    parsed = _parse_result_pos_neg(result)
    if parsed is None:
        return None
    query, positives, negatives = parsed
    if not positives or not negatives:
        return None
    return {
        "prompt": prompt,
        "query": query,
        "pos": positives,
        "neg": negatives,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Annotated_Main JSON -> MSMARCO JSONL with score-3 positives only"
    )
    parser.add_argument("--input", type=Path, required=True, help="Annotated JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--prompt", required=True, help="Retrieval instruction prompt")
    args = parser.parse_args()

    if not args.input.is_file():
        logger.error("input not found: %s", args.input)
        sys.exit(1)

    with args.input.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    results = data.get("results")
    if not isinstance(results, list):
        logger.error("missing or invalid 'results' array")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    score3_pos_empty = 0
    neg_empty = 0
    with args.output.open("w", encoding="utf-8") as output_file:
        for item in results:
            if not isinstance(item, dict):
                skipped += 1
                continue
            parsed = _parse_result_pos_neg(item)
            if parsed is None:
                skipped += 1
                continue
            query, positives, negatives = parsed
            if not positives:
                score3_pos_empty += 1
            if not negatives:
                neg_empty += 1
            if not positives or not negatives:
                skipped += 1
                continue
            record = {
                "prompt": args.prompt,
                "query": query,
                "pos": positives,
                "neg": negatives,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info(
        "done: results=%s written=%s skipped=%s score3_pos_empty=%s "
        "neg_empty=%s -> %s",
        len(results),
        written,
        skipped,
        score3_pos_empty,
        neg_empty,
        args.output,
    )


if __name__ == "__main__":
    main()
