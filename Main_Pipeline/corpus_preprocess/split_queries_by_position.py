#!/usr/bin/env python3
"""Split a pipeline JSON result so each document retains one query position."""

import argparse
import copy
import json
from pathlib import Path


def split_payload_by_query_position(payload, query_position):
    """Return a deep-copied payload retaining one 1-based query position."""
    if query_position < 1:
        raise ValueError("query_position must be at least 1")

    split_payload = copy.deepcopy(payload)
    retained_results = []
    for item in split_payload.get("results", []):
        queries = item.get("queries", [])
        if len(queries) < query_position:
            continue
        item["queries"] = [queries[query_position - 1]]
        retained_results.append(item)
    split_payload["results"] = retained_results
    split_payload["num_queries"] = 1
    return split_payload


def main():
    parser = argparse.ArgumentParser(
        description="Keep exactly one query position per document in a pipeline JSON result."
    )
    parser.add_argument("--input", required=True, help="Source DiverseQuery or Annotated JSON")
    parser.add_argument("--output", required=True, help="Destination JSON path")
    parser.add_argument(
        "--query-position",
        type=int,
        required=True,
        help="1-based query position to retain",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        payload = json.load(handle)
    split_payload = split_payload_by_query_position(payload, args.query_position)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
