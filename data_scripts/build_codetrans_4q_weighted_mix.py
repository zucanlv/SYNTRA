#!/usr/bin/env python3
"""Build a deterministic q1-q4 mixture for codetrans-dl continuation training."""

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-weight", type=int, default=10)
    parser.add_argument("--tail-weight", type=int, default=1)
    parser.add_argument("--max-per-query", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    if args.head_weight < 0 or args.tail_weight < 0:
        raise ValueError("mixture weights must be non-negative")
    if args.head_weight == 0 and args.tail_weight == 0:
        raise ValueError("at least one mixture weight must be positive")

    records: list[dict] = []
    counts: dict[str, int] = {}
    for query_index in range(1, 5):
        source = args.data_root / (
            f"codetrans-dl-4q-query{query_index}-20260812_"
            "exclude_other_pos_save_non_pos.jsonl"
        )
        rows = load_jsonl(source)
        if args.max_per_query > 0 and len(rows) > args.max_per_query:
            rows = random.Random(args.seed + query_index).sample(
                rows, args.max_per_query
            )
        weight = args.head_weight if query_index <= 2 else args.tail_weight
        records.extend(rows * weight)
        counts[f"q{query_index}"] = len(rows) * weight

    random.Random(args.seed).shuffle(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {"output": str(args.output), "total": len(records), "counts": counts},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
