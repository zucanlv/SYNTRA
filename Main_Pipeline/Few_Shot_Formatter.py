"""
Few-Shot Example Utilities
==========================
Centralised lookup and text-rendering for few-shot examples stored in
``Few_Shot_Example.py``.

Two public functions
--------------------
get_examples(task_name)
    Case-insensitive lookup that always returns a list (never raises).

format_examples(examples, max_examples=None)
    Converts a list of example dicts to a formatted prompt-ready string.
    Automatically handles both formats found in the data:

    - Triplet  : {"query": ..., "Positive": ..., "Hard Negative": ...}
    - Binary   : {"query": ..., "Positive": ...}          (no Hard Negative)

    Hard Negative lines are included only when the field is present and
    non-empty, so both formats render cleanly without stray empty lines.
"""

from typing import Dict, List, Optional

from Few_Shot_Example import Few_Shot_Example


def get_examples(task_name: str) -> List[Dict]:
    """Return few-shot examples for *task_name*, normalising case.

    Keys in ``Few_Shot_Example`` are lowercase (e.g. ``"msmarco"``).
    This function accepts any casing (``"MSMarco"``, ``"MSMARCO"``, …)
    and performs a lowercase lookup, returning an empty list on miss.

    Parameters
    ----------
    task_name : str
        Task identifier, case-insensitive.

    Returns
    -------
    list of dict
        The raw example dicts from ``Few_Shot_Example``, or ``[]``.
    """
    return Few_Shot_Example.get(task_name.lower(), [])


def format_examples(
    examples: List[Dict],
    max_examples: Optional[int] = None,
) -> str:
    """Render few-shot examples as a prompt-ready text block.

    Each example is numbered and rendered as:

        Example N:
        - Query: <query>
        - Positive document: <positive>
        - Hard Negative document: <hard_negative>   ← only if present

    The Hard Negative line is emitted only when ``ex["Hard Negative"]``
    exists and is non-empty, supporting both triplet and binary formats
    transparently.

    Parameters
    ----------
    examples : list of dict
        Raw example dicts (from ``get_examples`` or ``Few_Shot_Example``).
    max_examples : int, optional
        Cap the number of examples rendered. ``None`` means render all.

    Returns
    -------
    str
        Formatted text block, or ``""`` if *examples* is empty.
    """
    if not examples:
        return ""

    subset = examples[:max_examples] if max_examples is not None else examples

    blocks: List[str] = []
    for i, ex in enumerate(subset, 1):
        query = ex.get("query", "")
        positive = ex.get("Positive", "")
        hard_neg = ex.get("Hard Negative", "")

        lines = [
            f"Example {i}:",
            f"- Query: {query}",
            f"- Positive document: {positive}",
        ]
        if hard_neg.strip():
            lines.append(f"- Hard Negative document: {hard_neg}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def format_examples_queries_only(
    examples: List[Dict],
    max_examples: Optional[int] = None,
) -> str:
    """Render only the queries from few-shot examples (no documents).

    Useful when the prompt needs the gold-standard query list for style
    comparison but should avoid long document text. Each line is numbered.

    Parameters
    ----------
    examples : list of dict
        Raw example dicts (from ``get_examples`` or ``Few_Shot_Example``).
    max_examples : int, optional
        Cap the number of examples. ``None`` means use all.

    Returns
    -------
    str
        Formatted block of queries only, or ``""`` if *examples* is empty.
    """
    if not examples:
        return ""
    subset = examples[:max_examples] if max_examples is not None else examples
    lines = [f"Query {i}: {ex.get('query', '').strip()}" for i, ex in enumerate(subset, 1)]
    return "\n\n".join(lines)
