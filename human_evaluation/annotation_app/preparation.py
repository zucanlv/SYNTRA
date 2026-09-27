"""Administrator-only preparation of blinded experiment samples."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sampling import blind_samples, iter_source_pairs, reservoir_sample


ANSWER_KEY_FIELDS = (
    "sample_id",
    "dataset",
    "query",
    "document",
    "source_doc_id",
    "llm_score",
    "llm_classification",
    "llm_reasoning",
    "retrieval_score",
    "retrieval_rank",
    "is_original",
    "source_result_index",
    "source_query_index",
)


@dataclass(frozen=True)
class DatasetSource:
    """One public dataset name and its private annotated source file."""

    dataset: str
    path: Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_experiment(
    sources: list[DatasetSource],
    *,
    sample_count: int,
    seed: int,
    public_path: str | Path,
    private_dir: str | Path,
) -> dict[str, Any]:
    """Sample all sources and write blinded data plus private audit artifacts."""
    if not sources:
        raise ValueError("at least one dataset source is required")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if len({source.dataset for source in sources}) != len(sources):
        raise ValueError("dataset names must be unique")

    selected_by_dataset: dict[str, list[dict[str, Any]]] = {}
    manifest_datasets: list[dict[str, Any]] = []
    for offset, source in enumerate(sources):
        source_path = Path(source.path).resolve()
        derived_seed = seed + offset
        selected, seen = reservoir_sample(
            iter_source_pairs(source_path, source.dataset),
            sample_count,
            random.Random(derived_seed),
        )
        selected_by_dataset[source.dataset] = selected
        manifest_datasets.append(
            {
                "dataset": source.dataset,
                "source_path": str(source_path),
                "source_sha256": file_sha256(source_path),
                "derived_seed": derived_seed,
                "valid_pair_count": seen,
                "sample_count": len(selected),
            }
        )

    public, private = blind_samples(selected_by_dataset)
    public_output = Path(public_path)
    private_output = Path(private_dir)
    private_output.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        public_output,
        json.dumps(public, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_csv(private_output / "answer_key.csv", private)
    manifest = {
        "seed": seed,
        "sample_count_per_dataset": sample_count,
        "public_sample_count": len(public),
        "public_samples_sha256": file_sha256(public_output),
        "datasets": manifest_datasets,
    }
    _atomic_write_text(
        private_output / "sampling_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ANSWER_KEY_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in ANSWER_KEY_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
