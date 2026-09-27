"""
Split a training JSONL into median-pos-count split files.

For an input JSONL with entries shaped like:
  {"query": "...", "pos": [...], "neg": [...]}

This script:
  1. Computes x from the median number of positive documents per entry.
  2. Creates x split files under an output directory.
  3. Writes every input entry to every split, but keeps only one positive
     document in each split entry. Negative documents are unchanged.
  4. If the origin document from the diverse-query file appears in pos, it is
     moved to the first position before split selection, so split1 uses it.
  5. If an entry has fewer than x positives, split selection wraps around that
     entry's own pos list.

Example:
  python Main_Pipeline/corpus_preprocess/6-16-split-training-pos-by-median.py \
    --input-jsonl /path/to/training.jsonl \
    --diverse-query /path/to/DiverseQuery_Main.json \
    --output-dir /path/to/training_pos_splits
"""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand a training JSONL into split files by median pos count."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Input training-set JSONL path.",
    )
    parser.add_argument(
        "--diverse-query",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more DiverseQuery_Main_*.json or DiverseQuery_FromFile_*.json "
            "paths used for query -> origin doc mapping."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to input stem plus "
            "'_median_pos_splits' next to the input file."
        ),
    )
    parser.add_argument(
        "--median-rounding",
        choices=("floor", "ceil", "nearest"),
        default="ceil",
        help=(
            "How to convert a non-integer median into split count. "
            "Default: ceil."
        ),
    )
    parser.add_argument(
        "--split-count",
        type=int,
        default=None,
        help="Override the median-derived split count.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_num, json.loads(line)


def load_query_to_origin_doc(paths: list[Path]) -> dict[str, str]:
    query_to_origin_doc: dict[str, str] = {}

    for path in paths:
        print(f"Loading diverse-query file: {path}")
        with path.open("r", encoding="utf-8") as f:
            diverse_data = json.load(f)

        results = diverse_data.get("results", [])
        if not isinstance(results, list):
            raise ValueError(f"missing or invalid 'results' array: {path}")

        for result in results:
            if not isinstance(result, dict):
                continue
            origin_doc = result.get("doc", "")
            if not origin_doc:
                continue
            for q_obj in result.get("queries", []):
                if not isinstance(q_obj, dict):
                    continue
                query_text = q_obj.get("query", "")
                if query_text:
                    query_to_origin_doc[query_text] = origin_doc

    return query_to_origin_doc


def validate_pos(entry: dict[str, Any], line_num: int) -> list[Any]:
    pos = entry.get("pos", [])
    if not isinstance(pos, list):
        raise ValueError(f"line {line_num}: 'pos' must be a list")
    if not pos:
        raise ValueError(f"line {line_num}: 'pos' is empty")
    return pos


def collect_pos_counts(path: Path) -> list[int]:
    pos_counts = []
    for line_num, entry in iter_jsonl(path):
        pos_counts.append(len(validate_pos(entry, line_num)))
    if not pos_counts:
        raise ValueError(f"input JSONL has no valid entries: {path}")
    return pos_counts


def choose_split_count(pos_counts: list[int], rounding: str) -> tuple[int, float]:
    median_value = float(statistics.median(pos_counts))

    if rounding == "floor":
        split_count = math.floor(median_value)
    elif rounding == "ceil":
        split_count = math.ceil(median_value)
    else:
        split_count = int(median_value + 0.5)

    if split_count < 1:
        raise ValueError(f"median pos count produced invalid split count: {split_count}")

    return split_count, median_value


def move_origin_doc_to_first(pos: list[Any], origin_doc: str | None) -> tuple[list[Any], bool]:
    if not origin_doc or origin_doc not in pos:
        return list(pos), False

    reordered = [origin_doc]
    reordered.extend(doc for doc in pos if doc != origin_doc)
    return reordered, True


def write_splits(
    input_jsonl: Path,
    output_dir: Path,
    split_count: int,
    query_to_origin_doc: dict[str, str],
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / f"split{i}.jsonl" for i in range(1, split_count + 1)]

    stats = {
        "total": 0,
        "origin_moved_to_first": 0,
        "query_not_in_mapping": 0,
        "origin_not_in_pos": 0,
        "wrapped_entries": 0,
    }

    with (
        output_paths[0].open("w", encoding="utf-8") as first_fout,
    ):
        remaining_fouts = [
            path.open("w", encoding="utf-8") for path in output_paths[1:]
        ]
        fouts = [first_fout, *remaining_fouts]
        try:
            for line_num, entry in iter_jsonl(input_jsonl):
                stats["total"] += 1

                query = entry.get("query", "")
                origin_doc = query_to_origin_doc.get(query)
                if not origin_doc:
                    stats["query_not_in_mapping"] += 1

                pos = validate_pos(entry, line_num)
                ordered_pos, moved = move_origin_doc_to_first(pos, origin_doc)
                if moved:
                    stats["origin_moved_to_first"] += 1
                elif origin_doc:
                    stats["origin_not_in_pos"] += 1

                if len(ordered_pos) < split_count:
                    stats["wrapped_entries"] += 1

                for split_idx, fout in enumerate(fouts):
                    pos_idx = split_idx if len(ordered_pos) >= split_count else split_idx % len(ordered_pos)
                    out_entry = dict(entry)
                    out_entry["pos"] = [ordered_pos[pos_idx]]
                    fout.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
        finally:
            for fout in remaining_fouts:
                fout.close()

    return stats


def main() -> None:
    args = parse_args()

    if not args.input_jsonl.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")
    for path in args.diverse_query:
        if not path.is_file():
            raise FileNotFoundError(f"Diverse-query file not found: {path}")

    output_dir = args.output_dir or (
        args.input_jsonl.parent / f"{args.input_jsonl.stem}_median_pos_splits"
    )

    query_to_origin_doc = load_query_to_origin_doc(args.diverse_query)
    print(f"  Loaded {len(query_to_origin_doc):,} query -> origin-doc mappings")

    pos_counts = collect_pos_counts(args.input_jsonl)
    median_value = float(statistics.median(pos_counts))
    if args.split_count is None:
        split_count, _ = choose_split_count(pos_counts, args.median_rounding)
    else:
        split_count = args.split_count
        if split_count < 1:
            raise ValueError(f"--split-count must be >= 1, got: {split_count}")

    print(f"\nProcessing JSONL file: {args.input_jsonl}")
    print(f"  Entries              : {len(pos_counts):,}")
    print(f"  Median pos count     : {median_value:g}")
    print(f"  Median rounding      : {args.median_rounding}")
    print(f"  Split count          : {split_count:,}")
    if args.split_count is not None:
        print("  Split source         : --split-count override")
    else:
        print("  Split source         : median")

    stats = write_splits(
        input_jsonl=args.input_jsonl,
        output_dir=output_dir,
        split_count=split_count,
        query_to_origin_doc=query_to_origin_doc,
    )

    print(f"\n{'-'*60}")
    print(f"  Total entries read                  : {stats['total']:,}")
    print(f"  Split files written                 : {split_count:,}")
    print(f"  Entries per split                   : {stats['total']:,}")
    print(f"  Origin doc moved to first pos       : {stats['origin_moved_to_first']:,}")
    print(f"  Query not in mapping                : {stats['query_not_in_mapping']:,}")
    print(f"  Origin doc not in pos               : {stats['origin_not_in_pos']:,}")
    print(f"  Entries using wrapped pos selection : {stats['wrapped_entries']:,}")
    print(f"  Output directory                    : {output_dir}")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
