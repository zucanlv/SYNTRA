"""Validate and merge completed annotation submissions."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .storage import AnnotationDataError, AnnotationStore, validate_annotator_id


class ResultValidationError(ValueError):
    """Raised when submitted annotation files cannot be safely merged."""


def validate_submission(
    path: str | Path, samples: list[dict[str, Any]]
) -> tuple[str, dict[str, int]]:
    """Validate one complete submission and return its raw scores."""
    submission = Path(path)
    annotator_id = submission.stem
    try:
        validate_annotator_id(annotator_id)
        store = AnnotationStore(samples, submission, annotator_id)
    except (AnnotationDataError, ValueError) as exc:
        raise ResultValidationError(f"invalid submission {submission}: {exc}") from exc
    scores = store.scores()
    expected_ids = {str(sample["sample_id"]) for sample in samples}
    missing = expected_ids - set(scores)
    if missing:
        raise ResultValidationError(
            f"incomplete submission {submission}: missing {len(missing)} samples"
        )
    return annotator_id, scores


def merge_annotations(
    input_paths: Iterable[str | Path],
    samples: list[dict[str, Any]],
    output_path: str | Path,
    *,
    expected_count: int = 3,
) -> dict[str, Any]:
    """Merge complete submissions without deriving a consensus label."""
    paths = [Path(path) for path in input_paths]
    if len(paths) != expected_count:
        raise ResultValidationError(
            f"expected {expected_count} submissions, received {len(paths)}"
        )
    submissions: dict[str, dict[str, int]] = {}
    for path in paths:
        annotator_id, scores = validate_submission(path, samples)
        if annotator_id in submissions:
            raise ResultValidationError(f"duplicate annotator_id: {annotator_id}")
        submissions[annotator_id] = scores

    annotators = sorted(submissions)
    fields = ["sample_id", "dataset", *annotators]
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "dataset": sample["dataset"],
        }
        for annotator_id in annotators:
            row[annotator_id] = submissions[annotator_id][sample_id]
        rows.append(row)
    _atomic_write_csv(Path(output_path), fields, rows)
    return {
        "annotator_ids": annotators,
        "annotator_count": len(annotators),
        "sample_count": len(samples),
        "output_path": str(Path(output_path)),
    }


def _atomic_write_csv(
    path: Path, fields: list[str], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
