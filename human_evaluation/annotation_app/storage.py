"""Durable CSV storage and public-sample validation."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any


CSV_FIELDS = ("annotator_id", "sample_id", "dataset", "score")
PUBLIC_SAMPLE_FIELDS = {"sample_id", "dataset", "query", "document"}
ANNOTATOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class AnnotationDataError(ValueError):
    """Raised when samples or an existing annotation file are unsafe."""


def validate_annotator_id(value: str) -> str:
    """Return a safe annotator ID or raise ``ValueError``."""
    if not isinstance(value, str) or not ANNOTATOR_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "annotator ID must use 1-64 ASCII letters, numbers, '_' or '-', "
            "and must start with a letter or number"
        )
    return value


def load_samples(path: str | Path) -> list[dict[str, str]]:
    """Load a blinded sample list and reject extra fields or duplicate IDs."""
    sample_path = Path(path)
    try:
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationDataError(f"cannot load samples: {exc}") from exc
    if not isinstance(payload, list):
        raise AnnotationDataError("samples must be a JSON list")

    samples: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != PUBLIC_SAMPLE_FIELDS:
            raise AnnotationDataError(
                f"sample {index} must contain exactly these fields: "
                f"{sorted(PUBLIC_SAMPLE_FIELDS)}"
            )
        normalized: dict[str, str] = {}
        for field in PUBLIC_SAMPLE_FIELDS:
            value = item[field]
            if not isinstance(value, str) or not value.strip():
                raise AnnotationDataError(
                    f"sample {index} field {field!r} must be a non-empty string"
                )
            normalized[field] = value
        sample_id = normalized["sample_id"]
        if sample_id in seen:
            raise AnnotationDataError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        samples.append(normalized)
    return samples


class AnnotationStore:
    """Store one annotator's scores in an atomically replaced CSV file."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        output_path: str | Path,
        annotator_id: str,
    ) -> None:
        self.annotator_id = validate_annotator_id(annotator_id)
        self.output_path = Path(output_path)
        self._sample_order: list[str] = []
        self._samples: dict[str, dict[str, Any]] = {}
        for sample in samples:
            sample_id = str(sample["sample_id"])
            if sample_id in self._samples:
                raise AnnotationDataError(f"duplicate sample_id: {sample_id}")
            self._sample_order.append(sample_id)
            self._samples[sample_id] = sample
        self._scores: dict[str, int] = {}
        self._lock = threading.RLock()
        if self.output_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            with self.output_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                    raise AnnotationDataError(
                        f"annotation CSV header must be {','.join(CSV_FIELDS)}"
                    )
                for line_number, row in enumerate(reader, start=2):
                    self._load_row(row, line_number)
        except (OSError, csv.Error) as exc:
            raise AnnotationDataError(f"cannot load annotation CSV: {exc}") from exc

    def _load_row(self, row: dict[str, str], line_number: int) -> None:
        if row["annotator_id"] != self.annotator_id:
            raise AnnotationDataError(
                f"line {line_number} annotator_id does not match this session"
            )
        sample_id = row["sample_id"]
        sample = self._samples.get(sample_id)
        if sample is None:
            raise AnnotationDataError(f"line {line_number} has unknown sample_id")
        if sample_id in self._scores:
            raise AnnotationDataError(f"line {line_number} duplicates sample_id")
        if row["dataset"] != sample["dataset"]:
            raise AnnotationDataError(f"line {line_number} has wrong dataset")
        try:
            score = int(row["score"])
        except (TypeError, ValueError) as exc:
            raise AnnotationDataError(f"line {line_number} has invalid score") from exc
        if score not in {0, 1, 2, 3} or row["score"] != str(score):
            raise AnnotationDataError(f"line {line_number} has invalid score")
        self._scores[sample_id] = score

    def scores(self) -> dict[str, int]:
        """Return a snapshot of saved scores."""
        with self._lock:
            return dict(self._scores)

    def save(self, sample_id: str, score: int) -> None:
        """Save one score and atomically replace the on-disk CSV."""
        if type(score) is not int or score not in {0, 1, 2, 3}:
            raise ValueError("score must be one of 0, 1, 2, or 3")
        if sample_id not in self._samples:
            raise ValueError(f"unknown sample_id: {sample_id}")
        with self._lock:
            self._scores[sample_id] = score
            self._write_atomic()

    def _write_atomic(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            dir=self.output_path.parent,
            prefix=f".{self.output_path.name}.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for sample_id in self._sample_order:
                    if sample_id not in self._scores:
                        continue
                    writer.writerow(
                        {
                            "annotator_id": self.annotator_id,
                            "sample_id": sample_id,
                            "dataset": self._samples[sample_id]["dataset"],
                            "score": self._scores[sample_id],
                        }
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.output_path)
            self._fsync_parent_directory()
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def _fsync_parent_directory(self) -> None:
        flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
        directory_fd = os.open(self.output_path.parent, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
