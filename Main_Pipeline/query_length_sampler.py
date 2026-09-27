"""
Query Length Sampler
====================
Analyzes golden query length distribution from few-shot examples and provides
per-generation-call length constraint hints so that synthesized query lengths
align with the golden distribution rather than drifting toward the upper end
of any static range in the query instruction.

Typical usage (main.py):
    sampler = QueryLengthSampler.from_cache_or_generate(
        task_name=task,
        cache_dir=_INSTRUCTIONS_STORE_DIR,
        llm_cfg=None,          # None → fast statistical path (no extra LLM call)
    )
    # Inside _generate_dq_for_one, once per document:
    effective_instruction = query_instruction + sampler.sample_length_hint()

Design notes
------------
- ``from_cache_or_generate`` is called **once** per pipeline run and writes a
  ``QueryLengthDist_{task}_{timestamp}.json`` companion file alongside the
  existing ``QueryInstrRefine_*.txt`` files.
- ``sample_length_hint`` is called **per document** and is pure-Python
  (random.choices) — zero LLM cost at generation time.
- Two construction paths:
    statistical (default) — word-count percentiles from few-shot examples.
    LLM-assisted (opt-in)  — one LLM call for data+task-context-informed bucket
                             boundaries; falls back to statistical on error.
"""

from __future__ import annotations

import json
import logging
import random
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from Few_Shot_Formatter import get_examples
from HighLevel_Def import HighLevel_Task_Definition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LengthBucket:
    """One length category used for sampling."""
    label:       str            # "short" | "medium" | "long"
    word_min:    int
    word_max:    Optional[int]  # None = no upper bound
    probability: float          # empirical fraction; all buckets sum to 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """Number of whitespace-separated tokens in *text*."""
    return len(text.split())


def _compute_statistical_buckets(queries: List[str]) -> List[LengthBucket]:
    """Derive 3 length buckets from query word counts using p25/p75 quantiles.

    Handles edge cases gracefully:
    - Very few examples (< 2)
    - All queries have the same word count
    - p25 == p75 (degenerate distribution)
    """
    counts = [_word_count(q) for q in queries if q.strip()]
    if not counts:
        return [LengthBucket("medium", 1, None, 1.0)]

    n = len(counts)
    sorted_counts = sorted(counts)
    min_wc = sorted_counts[0]
    max_wc = sorted_counts[-1]

    if n == 1:
        return [LengthBucket(
            "medium",
            max(1, min_wc - 1),
            min_wc + 1,
            1.0,
        )]

    # Use statistics.quantiles (Python ≥ 3.8); fall back to index arithmetic
    try:
        q1, _, q3 = statistics.quantiles(sorted_counts, n=4)
    except (statistics.StatisticsError, ValueError):
        q1 = sorted_counts[n // 4]
        q3 = sorted_counts[(3 * n) // 4]

    lo = int(q1)  # inclusive upper bound for "short"
    hi = int(q3)  # inclusive upper bound for "medium"

    # Degenerate: all counts are the same or lo >= hi → collapse to 2 buckets
    if lo >= hi:
        med = sorted_counts[n // 2]
        lo  = max(1, med - 1)
        hi  = med

    # Empirical probabilities
    c_short  = sum(1 for w in counts if w <= lo)
    c_medium = sum(1 for w in counts if lo < w <= hi)
    c_long   = sum(1 for w in counts if w > hi)

    buckets: List[LengthBucket] = []
    if c_short > 0:
        buckets.append(LengthBucket(
            label       = "short",
            word_min    = min_wc,
            word_max    = lo,
            probability = c_short / n,
        ))
    if c_medium > 0:
        buckets.append(LengthBucket(
            label       = "medium",
            word_min    = lo + 1,
            word_max    = hi,
            probability = c_medium / n,
        ))
    if c_long > 0:
        buckets.append(LengthBucket(
            label       = "long",
            word_min    = hi + 1,
            word_max    = None,
            probability = c_long / n,
        ))

    if not buckets:
        # All queries identical length → single bucket
        return [LengthBucket("medium", min_wc, min_wc, 1.0)]

    # Renormalize to exactly 1.0
    total = sum(b.probability for b in buckets)
    for b in buckets:
        b.probability = round(b.probability / total, 6)
    return buckets


def _build_llm_analysis_prompt(task_name: str, queries: List[str]) -> str:
    task_def = HighLevel_Task_Definition.get(task_name, {})
    if isinstance(task_def, (list, tuple)) and len(task_def) == 3:
        query_type, doc_type, relevance_def = task_def
    else:
        query_type = relevance_def = task_name
        doc_type = "documents"

    word_counts = [_word_count(q) for q in queries if q.strip()]
    clean_queries = [q for q in queries if q.strip()]
    annotated = "\n".join(
        f"  {i + 1}. \"{q.strip()}\"  ({wc} words)"
        for i, (q, wc) in enumerate(zip(clean_queries, word_counts))
    )

    # Descriptive statistics surfaced in the prompt so the LLM reasons over data, not intuition
    if word_counts:
        wc_sorted = sorted(word_counts)
        n = len(wc_sorted)
        wc_min  = wc_sorted[0]
        wc_max  = wc_sorted[-1]
        wc_mean = round(sum(wc_sorted) / n, 1)
        mid     = n // 2
        wc_med  = wc_sorted[mid] if n % 2 == 1 else round((wc_sorted[mid - 1] + wc_sorted[mid]) / 2, 1)
        stats_line = (
            f"n={n} queries | min={wc_min} | max={wc_max} | "
            f"mean={wc_mean} | median={wc_med} words"
        )
    else:
        stats_line = "n=0 queries"

    return f"""### Role
You are an expert Corpus Linguist and IR Data Analyst specialising in query length distribution analysis for information retrieval data synthesis pipelines. Your analysis will be embedded directly into a query generation system, so every decision must be precise, empirically grounded, and immediately actionable by a downstream LLM.

### Task Context
- Task Name: {task_name}
- Query Type: {query_type}
- Document Type: {doc_type}
- Relevance Definition: {relevance_def}
- Purpose: Analyse the word-count distribution of the golden (reference) queries below and partition it into exactly 3 length categories. The resulting word-range boundaries and empirical probabilities will steer a query generation LLM so that synthesised query lengths mirror the golden distribution rather than drift toward the extremes of any static word-range.
- Downstream consumer: Only `word_min`, `word_max`, and `probability` are used downstream. No semantic description is needed — output purely numerical boundaries.

### Golden Query Samples (with word counts)
Distribution summary: {stats_line}

{annotated}

### Your Task — Two Phases

**Phase 1 — Integrated Distribution Analysis (reason step-by-step before producing output):**
1. Ground yourself in the Task Context first: given the Query Type, Document Type, and Relevance Definition above, form an expectation of what "short", "medium", and "long" naturally mean for this specific task — e.g. a keyword lookup task will have a very different length intuition than a detailed biomedical research query task.
2. Examine the golden query samples: review the actual queries and their word counts. Note the minimum, maximum, and any visible clustering or density gaps in the distribution.
3. Cross-validate expectation against data: do the natural clusters in the data align with your task-context intuition? Note any mismatches — e.g. if the task suggests long queries but the data is uniformly short, the data takes precedence; if the data has an ambiguous gap, use the task-context intuition to resolve it.
4. Identify the two breakpoints that best partition the data, informed by both the empirical density gaps and the task-level semantics of what constitutes a meaningfully different query length for this task.
5. Count how many samples fall into each of your three proposed partitions to derive the empirical probabilities.

**Phase 2 — Bucket Definition (the required JSON output):**
Based on Phase 1, define exactly three length buckets — "short", "medium", "long" — satisfying all of the following:
1. `word_min` / `word_max` boundaries reflect the breakpoints jointly determined by the empirical distribution and the task context in Phase 1.
2. `probability` is the empirical fraction of the golden samples that fall into that bucket — count directly from the data above. Probabilities must sum to exactly 1.0.

### Hard Constraints
- Define **exactly 3 buckets**, one per label: "short", "medium", "long" — in that order. Do not merge or omit any label, even if a bucket is sparsely populated.
- Probabilities must reflect genuine empirical counts from the samples above. Do not smooth, fabricate, or assume a uniform prior.
- Return a **single valid JSON object**. No Markdown fences, no preamble, no commentary, no extra keys.

### Output Format
{{
  "buckets": [
    {{
      "label": "short",
      "word_min": <integer>,
      "word_max": <integer>,
      "probability": <float 0–1, empirically derived>
    }},
    {{
      "label": "medium",
      "word_min": <integer>,
      "word_max": <integer>,
      "probability": <float 0–1, empirically derived>
    }},
    {{
      "label": "long",
      "word_min": <integer>,
      "word_max": <integer or null>,
      "probability": <float 0–1, empirically derived>
    }}
  ]
}}\
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QueryLengthSampler:
    """Provides per-call query length constraint hints for DiverseQueryGenerator.

    Build once per pipeline run via ``from_cache_or_generate``, then call
    ``sample_length_hint()`` inside ``_generate_dq_for_one`` before each
    DiverseQuery tool call.  The returned string is appended to
    ``task_instruction``; no changes to ``DiverseQueryGenerator``'s interface
    are required.
    """

    def __init__(self, buckets: List[LengthBucket], task_name: str) -> None:
        if not buckets:
            raise ValueError("buckets must be non-empty")
        self.buckets   = buckets
        self.task_name = task_name

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_examples_statistical(cls, task_name: str) -> "QueryLengthSampler":
        """Statistical analysis of few-shot golden queries — no LLM cost."""
        examples = get_examples(task_name)
        queries  = [ex.get("query", "") for ex in examples]
        buckets  = _compute_statistical_buckets(queries)
        logger.info(
            "QueryLengthSampler [statistical] task='%s' buckets=%s",
            task_name,
            [(b.label, f"{b.word_min}-{b.word_max}", f"p={b.probability:.2f}") for b in buckets],
        )
        return cls(buckets, task_name)

    @classmethod
    def from_examples_with_llm(
        cls, task_name: str, llm_cfg: Dict
    ) -> "QueryLengthSampler":
        """LLM-assisted analysis for data+task-context-informed bucket boundaries.

        Falls back to the statistical path if the LLM call fails or returns
        unparseable output, so the pipeline is never blocked.
        """
        import json5
        from qwen_agent.agents import Assistant

        examples = get_examples(task_name)
        queries  = [ex.get("query", "") for ex in examples]
        if not queries:
            logger.warning(
                "No few-shot examples for task '%s'; using statistical fallback.", task_name
            )
            return cls.from_examples_statistical(task_name)

        prompt = _build_llm_analysis_prompt(task_name, queries)
        agent  = Assistant(llm=llm_cfg)
        raw    = ""
        try:
            for response in agent.run([{"role": "user", "content": prompt}]):
                raw = response[-1]["content"]
            parsed   = json5.loads(raw)
            raw_bkts = parsed.get("buckets", [])
            buckets: List[LengthBucket] = []
            for b in raw_bkts:
                prob = float(b.get("probability", 0.0))
                if prob <= 0:
                    continue
                buckets.append(LengthBucket(
                    label       = str(b.get("label", "medium")),
                    word_min    = int(b["word_min"]) if b.get("word_min") is not None else 1,
                    word_max    = int(b["word_max"]) if b.get("word_max") is not None else None,
                    probability = prob,
                ))
            if not buckets:
                raise ValueError("LLM returned no buckets with positive probability")

            # Renormalize
            total = sum(b.probability for b in buckets)
            for b in buckets:
                b.probability = round(b.probability / total, 6)

            logger.info(
                "QueryLengthSampler [LLM] task='%s' buckets=%s",
                task_name,
                [(b.label, f"{b.word_min}-{b.word_max}", f"p={b.probability:.2f}") for b in buckets],
            )
            return cls(buckets, task_name)

        except Exception as exc:
            logger.warning(
                "LLM bucket analysis failed for task '%s' (%s); "
                "falling back to statistical analysis.",
                task_name, exc,
            )
            return cls.from_examples_statistical(task_name)

    @classmethod
    def from_cache_or_generate(
        cls,
        task_name: str,
        cache_dir: Path,
        llm_cfg:   Optional[Dict] = None,
    ) -> "QueryLengthSampler":
        """Load a cached ``QueryLengthDist_{task}_*.json`` or generate one.

        Parameters
        ----------
        task_name : str
        cache_dir : Path
            The same ``_INSTRUCTIONS_STORE_DIR`` used for ``QueryInstrRefine_*``
            files so all per-task artefacts stay together.
        llm_cfg : dict, optional
            When provided, attempt the LLM analysis path; otherwise use the
            fast statistical path.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(
            cache_dir.glob(f"QueryLengthDist_{task_name}_*.json"),
            reverse=True,
        )
        if existing:
            path = existing[0]
            logger.info("Loading cached QueryLengthDist from %s", path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data, task_name)

        # Generate
        sampler = (
            cls.from_examples_with_llm(task_name, llm_cfg)
            if llm_cfg
            else cls.from_examples_statistical(task_name)
        )

        # Persist alongside the other instruction files
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = cache_dir / f"QueryLengthDist_{task_name}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sampler.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Saved QueryLengthDist to %s", path)
        return sampler

    # ------------------------------------------------------------------
    # Sampling (hot path — called once per document)
    # ------------------------------------------------------------------

    def sample_length_hint(self) -> str:
        """Sample one length bucket and return a prompt-ready constraint string.

        The string is designed to be **appended** to ``task_instruction``
        (with no modifications to ``DiverseQueryGenerator``).

        The sampled bucket is chosen with weights equal to the empirical
        probabilities derived from few-shot golden queries, so across many
        documents the overall distribution of synthesized query lengths will
        mirror the golden distribution.

        Example output (short bucket sampled):
            \\n\\n**[Length Constraint — follow strictly for this generation batch]**
            Generate **short** queries (target 3–5 words per query).
            This overrides any conflicting length guidance above.
        """
        bucket = random.choices(  # noqa: S311
            self.buckets,
            weights=[b.probability for b in self.buckets],
        )[0]
        word_range = (
            f"{bucket.word_min}–{bucket.word_max} words"
            if bucket.word_max is not None
            else f"{bucket.word_min}+ words"
        )
        return (
            f"***IMPORTANT!!!:***\n"
            f"\n\n**[Length Constraint — follow strictly for this generation batch]**\n"
            f"Generate **{bucket.label}** queries (target {word_range} per query). "
            f"**This overrides any conflicting length guidance above.**"
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        examples = get_examples(self.task_name)
        queries  = [ex.get("query", "") for ex in examples]
        return {
            "task_name":      self.task_name,
            "source_queries": queries,
            "word_counts":    [_word_count(q) for q in queries if q.strip()],
            "buckets": [
                {
                    "label":       b.label,
                    "word_min":    b.word_min,
                    "word_max":    b.word_max,
                    "probability": b.probability,
                }
                for b in self.buckets
            ],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    @classmethod
    def from_dict(cls, data: Dict, task_name: str) -> "QueryLengthSampler":
        buckets = [
            LengthBucket(
                label       = b["label"],
                word_min    = int(b["word_min"]) if b.get("word_min") is not None else 1,
                word_max    = int(b["word_max"]) if b.get("word_max") is not None else None,
                probability = b["probability"],
            )
            for b in data.get("buckets", [])
            if b.get("probability", 0) > 0
        ]
        if not buckets:
            raise ValueError(
                f"No valid buckets (probability > 0) in cached data for task '{task_name}'"
            )
        return cls(buckets, task_name)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{b.label}(p={b.probability:.2f}, {b.word_min}–{b.word_max if b.word_max else '∞'} words)"
            for b in self.buckets
        )
        return f"QueryLengthSampler(task={self.task_name!r}, [{parts}])"
