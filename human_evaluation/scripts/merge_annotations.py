#!/usr/bin/env python3
"""Merge three completed CSV files without deriving ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation_app.results import merge_annotations
from annotation_app.storage import load_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and merge three annotators' raw 0-3 scores."
    )
    parser.add_argument("annotation_csv", nargs="+", type=Path)
    parser.add_argument(
        "--samples", type=Path, default=ROOT / "data" / "samples.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "merged_annotations.csv"
    )
    parser.add_argument("--expected-count", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    samples = load_samples(args.samples)
    summary = merge_annotations(
        args.annotation_csv,
        samples,
        args.output,
        expected_count=args.expected_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
