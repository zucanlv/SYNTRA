#!/usr/bin/env python3
"""Split a large Query2Passage JSON into valid smaller Query2Passage JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ijson


def _read_prefix_metadata(path: Path) -> dict[str, Any]:
    """Read top-level metadata that appears before the top-level results array."""
    meta: dict[str, Any] = {}
    with path.open("rb") as f:
        parser = ijson.parse(f, use_float=True)
        current_key: str | None = None
        for _prefix, event, value in parser:
            if event == "map_key":
                if value == "results":
                    break
                current_key = value
                continue
            if current_key and event in {"string", "number", "boolean", "null"}:
                meta[current_key] = value
                current_key = None
    return meta


def _count_results(path: Path) -> int:
    total = 0
    with path.open("rb") as f:
        for _item in ijson.items(f, "results.item", use_float=True):
            total += 1
            if total % 25000 == 0:
                print(f"counted results: {total}", flush=True)
    return total


def _write_header(handle, meta: dict[str, Any]) -> None:
    handle.write("{\n")
    for key, value in meta.items():
        handle.write("  ")
        json.dump(key, handle, ensure_ascii=False)
        handle.write(": ")
        json.dump(value, handle, ensure_ascii=False)
        handle.write(",\n")
    handle.write('  "results": [\n')


def _write_footer(handle) -> None:
    handle.write("\n  ]\n}\n")


def split_q2p(input_path: Path, output1: Path, output2: Path) -> tuple[int, int, int]:
    meta = _read_prefix_metadata(input_path)
    if not meta:
        print("warning: no top-level metadata found before results", flush=True)

    total = _count_results(input_path)
    if total <= 1:
        raise ValueError(f"need at least 2 results items to split, got {total}")
    split_at = (total + 1) // 2
    print(f"total results: {total}; split_at: {split_at}", flush=True)

    output1.parent.mkdir(parents=True, exist_ok=True)
    output2.parent.mkdir(parents=True, exist_ok=True)

    n1 = 0
    n2 = 0
    with input_path.open("rb") as src, output1.open("w", encoding="utf-8") as f1, output2.open(
        "w", encoding="utf-8"
    ) as f2:
        _write_header(f1, meta)
        _write_header(f2, meta)
        first1 = True
        first2 = True

        for idx, item in enumerate(ijson.items(src, "results.item", use_float=True)):
            if idx < split_at:
                if not first1:
                    f1.write(",\n")
                json.dump(item, f1, ensure_ascii=False, separators=(",", ":"))
                first1 = False
                n1 += 1
            else:
                if not first2:
                    f2.write(",\n")
                json.dump(item, f2, ensure_ascii=False, separators=(",", ":"))
                first2 = False
                n2 += 1

            done = idx + 1
            if done % 25000 == 0:
                print(f"written results: {done}/{total}", flush=True)

        _write_footer(f1)
        _write_footer(f2)

    return total, n1, n2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output1", type=Path, required=True)
    parser.add_argument("--output2", type=Path, required=True)
    args = parser.parse_args()

    total, n1, n2 = split_q2p(args.input, args.output1, args.output2)
    print(json.dumps({"total_results": total, "part1_results": n1, "part2_results": n2}, indent=2))


if __name__ == "__main__":
    main()
