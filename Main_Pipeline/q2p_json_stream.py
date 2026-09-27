"""Streaming helpers for very large Query2Passage JSON files.

Full-document ``json.load`` on multi-GB files can exhaust RAM (OOM).  These
utilities use ``ijson`` to validate, count, sample, or iterate ``results`` items
without building the entire tree.
"""

from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List

import ijson

# Files at or above this size use streaming / no full json.load (main, Hn_Mine, etc.).
LARGE_Q2P_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB


def file_is_large_q2p(path: str | Path) -> bool:
    try:
        return Path(path).stat().st_size >= LARGE_Q2P_BYTES
    except OSError:
        return False


def extract_input_path_prefix(path: str | Path, max_scan: int = 2_097_152) -> str:
    """Read ``input_path`` from the start of a Q2P JSON (it is written first)."""
    with open(path, "rb") as f:
        chunk = f.read(max_scan).decode("utf-8", errors="replace")
    m = re.search(r'"input_path"\s*:\s*("(?:[^"\\]|\\.)*")', chunk)
    if not m:
        return ""
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return ""


def _q2p_query_iter_from_item(item: Any) -> Iterator[Dict[str, Any]]:
    if not isinstance(item, dict):
        return
    for q_item in item.get("queries", []) or []:
        if isinstance(q_item, str):
            if not q_item.strip():
                continue
            yield {"query": q_item, "candidates": []}
        elif isinstance(q_item, dict) and q_item.get("query"):
            yield q_item


def validate_q2p_has_nonempty_candidates(path: str | Path) -> bool:
    """True if some query object has a non-empty ``candidates`` list.

    Matches ``materialize_step5_query2passage_input`` small-file checks: only
    ``dict`` entries are considered (same as ``for q_obj in item['queries']`` +
    ``isinstance(q_obj, dict)``), and ``query`` text is **not** required.
    """
    with open(path, "rb") as f:
        for item in ijson.items(f, "results.item"):
            if not isinstance(item, dict):
                continue
            for q_obj in item.get("queries", []) or []:
                if not isinstance(q_obj, dict):
                    continue
                cands = q_obj.get("candidates")
                if isinstance(cands, list) and len(cands) > 0:
                    return True
    return False


def count_queries_q2p_streaming(path: str | Path) -> int:
    """Same counting rules as ``count_queries_in_q2p_json`` on a dict."""
    n = 0
    with open(path, "rb") as f:
        for item in ijson.items(f, "results.item"):
            if not isinstance(item, dict):
                continue
            for q_item in item.get("queries", []) or []:
                if isinstance(q_item, str):
                    if q_item.strip():
                        n += 1
                elif isinstance(q_item, dict) and q_item.get("query"):
                    n += 1
    return n


def reservoir_sample_q_objs_with_candidates(
    path: str | Path,
    k: int,
    rng: Any = None,
) -> List[Dict[str, Any]]:
    """Reservoir sample up to ``k`` query dicts that have non-empty candidates.

    Returns **deep copies** safe to use after the streaming parse advances.
    ``rng`` may be ``random`` or ``random.Random`` (must provide ``randint``).
    """
    if k <= 0:
        return []
    r = rng or random
    pool: List[Dict[str, Any]] = []
    seen = 0
    with open(path, "rb") as f:
        for item in ijson.items(f, "results.item"):
            for q_obj in _q2p_query_iter_from_item(item):
                cands = q_obj.get("candidates") if isinstance(q_obj, dict) else None
                if not isinstance(cands, list) or len(cands) == 0:
                    continue
                if not isinstance(q_obj, dict) or not q_obj.get("query"):
                    continue
                seen += 1
                if len(pool) < k:
                    pool.append(copy.deepcopy(q_obj))
                else:
                    j = r.randint(1, seen)
                    if j <= k:
                        pool[j - 1] = copy.deepcopy(q_obj)
    return pool
