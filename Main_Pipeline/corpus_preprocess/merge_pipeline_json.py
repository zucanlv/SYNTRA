#!/usr/bin/env python3
"""Stream-merge pipeline JSON files by concatenating their ``results`` arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import ijson


def read_prefix_metadata(path: Path) -> dict[str, Any]:
    """Read scalar top-level fields that appear before ``results``."""
    metadata: dict[str, Any] = {}
    current_key: str | None = None
    with path.open("rb") as handle:
        for _prefix, event, value in ijson.parse(handle, use_float=True):
            if event == "map_key":
                if value == "results":
                    break
                current_key = value
            elif current_key and event in {"string", "number", "boolean", "null"}:
                metadata[current_key] = value
                current_key = None
    return metadata


def merge_pipeline_json(inputs: list[Path], output: Path) -> int:
    """Concatenate ``results`` in input order and return the item count."""
    if not inputs:
        raise ValueError("at least one input file is required")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"input file not found: {path}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    metadata = read_prefix_metadata(inputs[0])
    for path in inputs[1:]:
        other_metadata = read_prefix_metadata(path)
        if other_metadata != metadata:
            raise ValueError(
                f"top-level metadata mismatch: {inputs[0]} != {path}: "
                f"{metadata!r} != {other_metadata!r}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    total_results = 0
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            destination.write("{\n")
            for key, value in metadata.items():
                destination.write("  ")
                json.dump(key, destination, ensure_ascii=False)
                destination.write(": ")
                json.dump(value, destination, ensure_ascii=False)
                destination.write(",\n")
            destination.write('  "results": [\n')

            first_result = True
            for path in inputs:
                with path.open("rb") as source:
                    for item in ijson.items(source, "results.item", use_float=True):
                        if not first_result:
                            destination.write(",\n")
                        json.dump(item, destination, ensure_ascii=False, separators=(",", ":"))
                        first_result = False
                        total_results += 1

            destination.write("\n  ]\n}\n")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return total_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Input files in merge order")
    parser.add_argument("--output", required=True, type=Path, help="Merged output JSON")
    args = parser.parse_args()

    total_results = merge_pipeline_json(args.inputs, args.output)
    print(
        json.dumps(
            {
                "inputs": [str(path) for path in args.inputs],
                "output": str(args.output),
                "total_results": total_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
