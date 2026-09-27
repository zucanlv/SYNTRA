#!/usr/bin/env python3
"""Build qpos JSON files from reason-embed-data JSONL files.

The output matches the pipeline's Step 4 DiverseQuery/qpos input format:

{
  "input_path": ".../economics-formatted.jsonl",
  "num_queries": 1,
  "results": [
    {
      "doc_id": "economics-formatted_00000000",
      "doc": "sampled positive passage",
      "queries": [{"query": "query text"}]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = [
    Path(
        "/data/share/project/shared_datasets/reason-embed-data/"
        "reason-embed-data-0928/economics-formatted.jsonl"
    ),
    Path(
        "/data/share/project/shared_datasets/reason-embed-data/"
        "reason-embed-data-0928/pony-formatted.jsonl"
    ),
    Path(
        "/data/share/project/shared_datasets/reason-embed-data/"
        "reason-embed-data-0928/theoremqa_questions-formatted.jsonl"
    ),
]
DEFAULT_OUTPUT_DIR = Path(
    "/data/share/project/shared_datasets/DSA/others/qpos/reason-embed-data-0928"
)
DEFAULT_SEED = 20260629


def _valid_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def convert_one(input_path: Path, output_dir: Path, seed: int) -> dict[str, int | str]:
    if not input_path.is_file():
        raise FileNotFoundError(f"input JSONL not found: {input_path}")

    rng = random.Random(f"{seed}:{input_path.name}")
    results: list[dict[str, Any]] = []
    skipped_missing_query = 0
    skipped_missing_pos = 0
    rows_read = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rows_read += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {input_path}:{line_no}: {exc}") from exc

            query = _valid_text(obj.get("query"))
            if query is None:
                skipped_missing_query += 1
                continue

            pos = obj.get("pos")
            positives = []
            if isinstance(pos, list):
                positives = [text for text in (_valid_text(item) for item in pos) if text]
            if not positives:
                skipped_missing_pos += 1
                continue

            sampled_pos = rng.choice(positives)
            results.append(
                {
                    "doc_id": f"{input_path.stem}_{len(results):08d}",
                    "doc": sampled_pos,
                    "queries": [{"query": query}],
                }
            )

    if not results:
        raise ValueError(f"no usable query-positive rows found in {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_qpos.json"
    payload = {
        "input_path": str(input_path.resolve()),
        "num_queries": 1,
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows_read": rows_read,
        "results": len(results),
        "skipped_missing_query": skipped_missing_query,
        "skipped_missing_pos": skipped_missing_pos,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert reason-embed-data JSONL files to qpos JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        dest="inputs",
        help="Input JSONL path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for positive sampling. Default: {DEFAULT_SEED}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.inputs or DEFAULT_INPUTS
    summaries = [convert_one(path, args.output_dir, args.seed) for path in inputs]
    json.dump(summaries, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
