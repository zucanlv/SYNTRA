#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "corpus_preprocess/anno_queries_to_msmarco_syn_fill_easy.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("anno_fill_easy", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _result(hard_ids: list[str], easy_ids: list[str]) -> dict:
    candidates = [{"doc_id": "p1", "doc": "positive"}]
    candidates.extend({"doc_id": doc_id, "doc": f"text {doc_id}"} for doc_id in hard_ids)
    candidates.extend({"doc_id": doc_id, "doc": f"text {doc_id}"} for doc_id in easy_ids)
    return {
        "queries": [
            {
                "query": "sample query",
                "candidates": candidates,
                "positive_doc_ids": ["p1"],
                "hard_negative_doc_ids": hard_ids,
                "easy_negative_doc_ids": easy_ids,
            }
        ]
    }


def test_uses_all_hard_negatives_when_target_is_met():
    module = _load_module()
    rec = module.result_to_record(
        _result(["h1", "h2", "h3"], ["e1"]),
        prompt="prompt",
        target_negatives=2,
    )
    assert rec is not None
    assert rec["neg"] == ["text h1", "text h2", "text h3"]


def test_fills_hard_negatives_with_easy_negatives_to_target():
    module = _load_module()
    rec = module.result_to_record(
        _result(["h1", "h2"], ["e1", "e2", "e3"]),
        prompt="prompt",
        target_negatives=4,
    )
    assert rec is not None
    assert rec["neg"] == ["text h1", "text h2", "text e1", "text e2"]


def test_keeps_record_with_all_easy_negatives_when_target_cannot_be_reached():
    module = _load_module()
    rec = module.result_to_record(
        _result(["h1"], ["e1"]),
        prompt="prompt",
        target_negatives=3,
    )
    assert rec is not None
    assert rec["neg"] == ["text h1", "text e1"]


def test_skips_when_no_hard_or_easy_negatives():
    module = _load_module()
    rec = module.result_to_record(
        _result([], []),
        prompt="prompt",
        target_negatives=3,
    )
    assert rec is None


if __name__ == "__main__":
    test_uses_all_hard_negatives_when_target_is_met()
    test_fills_hard_negatives_with_easy_negatives_to_target()
    test_keeps_record_with_all_easy_negatives_when_target_cannot_be_reached()
    test_skips_when_no_hard_or_easy_negatives()
    print("ok")
