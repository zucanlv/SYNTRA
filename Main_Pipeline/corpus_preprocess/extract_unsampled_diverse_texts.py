#!/usr/bin/env python3
"""Extract records from a full JSONL corpus that are absent from a sample."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


DEFAULT_FULL = Path(
    "/data/share/project/shared_datasets/DSA/others/diverse_texts/"
    "diverse_texts_cos062_training_corpus_499996_93804.jsonl"
)
DEFAULT_SAMPLE = Path(
    "/data/share/project/shared_datasets/DSA/others/diverse_texts/"
    "diverse_texts_cos062_training_corpus_499996_93804_sample80k.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/data/share/project/shared_datasets/DSA/others/diverse_texts/"
    "diverse_texts_cos062_training_corpus_499996_93804_unsampled13804.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract full-corpus records whose doc_id is not in the sample."
    )
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=13_804,
        help="Fail unless this many unsampled records are found.",
    )
    return parser.parse_args()


def load_record(line: str, path: Path, line_number: int) -> dict:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"Expected JSON object at {path}:{line_number}")
    doc_id = record.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError(f"Missing or invalid doc_id at {path}:{line_number}")
    return record


def load_sample(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = load_record(line, path, line_number)
            doc_id = record["doc_id"]
            if doc_id in records:
                raise ValueError(f"Duplicate sample doc_id: {doc_id}")
            records[doc_id] = record
    return records


def main() -> None:
    args = parse_args()
    for path in (args.full, args.sample):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
    if args.expected_count < 0:
        raise ValueError("--expected-count must be non-negative")

    sample_records = load_sample(args.sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    full_ids: set[str] = set()
    unsampled_count = 0
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            with args.full.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    record = load_record(line, args.full, line_number)
                    doc_id = record["doc_id"]
                    if doc_id in full_ids:
                        raise ValueError(f"Duplicate full-corpus doc_id: {doc_id}")
                    full_ids.add(doc_id)

                    sample_record = sample_records.get(doc_id)
                    if sample_record is None:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        unsampled_count += 1
                    elif sample_record != record:
                        raise ValueError(
                            f"Record content differs between full and sample: {doc_id}"
                        )

        sample_only_ids = set(sample_records) - full_ids
        if sample_only_ids:
            example = next(iter(sample_only_ids))
            raise ValueError(
                f"Sample is not a subset of full corpus; example doc_id: {example}"
            )
        if unsampled_count != args.expected_count:
            raise ValueError(
                f"Expected {args.expected_count} unsampled records, "
                f"found {unsampled_count}"
            )

        os.replace(temp_path, args.output)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print(f"Full records      : {len(full_ids):,}")
    print(f"Sampled records   : {len(sample_records):,}")
    print(f"Unsampled records : {unsampled_count:,}")
    print(f"Output             : {args.output}")


if __name__ == "__main__":
    main()
