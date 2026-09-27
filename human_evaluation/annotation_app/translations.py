"""Validation and attachment of bundled sample translations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TRANSLATION_FIELDS = {"query", "document"}


class TranslationDataError(ValueError):
    """Raised when the bundled translation data is incomplete or malformed."""


def load_translations(
    path: str | Path, samples: list[dict[str, str]]
) -> dict[str, dict[str, str]]:
    """Load one query/document translation for every public sample."""
    translation_path = Path(path)
    try:
        payload = json.loads(translation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranslationDataError(f"cannot load translations: {exc}") from exc
    if not isinstance(payload, dict):
        raise TranslationDataError("translations must be a JSON object keyed by sample_id")

    sample_ids = {sample["sample_id"] for sample in samples}
    translation_ids = set(payload)
    if translation_ids != sample_ids:
        missing = sorted(sample_ids - translation_ids)
        extra = sorted(translation_ids - sample_ids)
        details = []
        if missing:
            details.append(f"missing sample IDs: {missing}")
        if extra:
            details.append(f"unknown sample IDs: {extra}")
        raise TranslationDataError("; ".join(details))

    translations: dict[str, dict[str, str]] = {}
    for sample_id, item in payload.items():
        if not isinstance(item, dict) or set(item) != TRANSLATION_FIELDS:
            raise TranslationDataError(
                f"translation {sample_id!r} must contain exactly "
                f"{sorted(TRANSLATION_FIELDS)}"
            )
        normalized: dict[str, str] = {}
        for field in TRANSLATION_FIELDS:
            value = item[field]
            if not isinstance(value, str) or not value.strip():
                raise TranslationDataError(
                    f"translation {sample_id!r} field {field!r} "
                    "must be a non-empty string"
                )
            normalized[field] = value
        translations[sample_id] = normalized
    return translations


def attach_translations(
    samples: list[dict[str, str]], translations: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    """Return API samples enriched with Chinese query/document text."""
    return [
        {
            **sample,
            "query_zh": translations[sample["sample_id"]]["query"],
            "document_zh": translations[sample["sample_id"]]["document"],
        }
        for sample in samples
    ]
