#!/usr/bin/env python3
"""Validate one completed annotation CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation_app.results import validate_submission
from annotation_app.storage import load_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that an annotation CSV has one valid score per sample."
    )
    parser.add_argument("annotation_csv", type=Path)
    parser.add_argument(
        "--samples", type=Path, default=ROOT / "data" / "samples.json"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    samples = load_samples(args.samples)
    annotator_id, scores = validate_submission(args.annotation_csv, samples)
    print(
        json.dumps(
            {
                "valid": True,
                "annotator_id": annotator_id,
                "sample_count": len(scores),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
