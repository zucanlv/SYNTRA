"""Shared production-mode sampling logic (Step 2.6 / preprocessing).

Used by main.py and corpus_preprocess scripts so random|range|indices behavior
matches config_msmarco production_sample settings.
"""

from __future__ import annotations

import random


def apply_production_sample(docs: list, cfg: dict, logger) -> list:
    """Sample a subset of docs for production-mode processing.

    Sampling modes (cfg["mode"]):
      random  – randomly pick cfg["count"] docs; optional cfg["seed"] for reproducibility.
      range   – slice by 0-based closed index interval cfg["range"] = "start-end".
      indices – pick specific 0-based row indices listed in cfg["indices"].

    The returned list preserves original dict objects (no deep copy).
    If configuration is invalid or count ≥ total, the full list is returned with a warning.
    """
    mode = (cfg.get("mode") or "random").strip().lower()
    total = len(docs)

    if mode == "random":
        count = cfg.get("count")
        if count is None:
            logger.info(
                "[ProductionSample] mode=random: count not set; using all %d docs (no subsampling)",
                total,
            )
            return docs
        count = int(count)
        if count >= total:
            logger.info(
                "[ProductionSample] mode=random: count=%d >= total=%d; using all docs",
                count,
                total,
            )
            return docs
        seed = cfg.get("seed")
        if seed is not None:
            random.seed(int(seed))
        sampled = random.sample(docs, count)
        logger.info(
            "[ProductionSample] mode=random: %d → %d docs (seed=%s)",
            total,
            len(sampled),
            seed,
        )
        return sampled

    if mode == "range":
        range_str = (cfg.get("range") or "").strip()
        parts = range_str.split("-", 1)
        if len(parts) != 2:
            logger.error(
                "[ProductionSample] mode=range: invalid range '%s' (expected 'start-end'); skipping",
                range_str,
            )
            return docs
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError:
            logger.error(
                "[ProductionSample] mode=range: cannot parse '%s' as integers; skipping",
                range_str,
            )
            return docs
        sampled = docs[start : end + 1]
        logger.info(
            "[ProductionSample] mode=range: %d → %d docs (range=%d-%d)",
            total,
            len(sampled),
            start,
            end,
        )
        return sampled

    if mode == "indices":
        raw = cfg.get("indices")
        if not raw:
            logger.warning("[ProductionSample] mode=indices: 'indices' not set; skipping sampling")
            return docs
        if isinstance(raw, str):
            idx_list = [int(x.strip()) for x in raw.split(",") if x.strip()]
        else:
            idx_list = [int(i) for i in raw]
        valid = [i for i in idx_list if 0 <= i < total]
        skipped = len(idx_list) - len(valid)
        if skipped:
            logger.warning(
                "[ProductionSample] mode=indices: %d indices out of range [0, %d); skipped",
                skipped,
                total,
            )
        sampled = [docs[i] for i in sorted(set(valid))]
        logger.info(
            "[ProductionSample] mode=indices: %d → %d docs",
            total,
            len(sampled),
        )
        return sampled

    logger.error("[ProductionSample] Unknown mode '%s'; skipping sampling", mode)
    return docs
