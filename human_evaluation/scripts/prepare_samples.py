#!/usr/bin/env python3
"""Prepare blinded samples and private audit files from annotated sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation_app.preparation import DatasetSource, prepare_experiment


def dataset_source(value: str) -> DatasetSource:
    try:
        dataset, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use DATASET=/path/to/source.json") from exc
    if not dataset or not path:
        raise argparse.ArgumentTypeError("use DATASET=/path/to/source.json")
    return DatasetSource(dataset, Path(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Uniformly sample annotated pairs and create blinded public data."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=dataset_source,
        metavar="NAME=PATH",
        help="Dataset key and Annotated_Main JSON path; repeat once per dataset",
    )
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--public-path", type=Path, default=ROOT / "data" / "samples.json"
    )
    parser.add_argument("--private-dir", type=Path, default=ROOT / "private")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = prepare_experiment(
        args.dataset,
        sample_count=args.sample_count,
        seed=args.seed,
        public_path=args.public_path,
        private_dir=args.private_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
