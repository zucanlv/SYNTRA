"""
Script: 4-29-exclude-other-pos-save-non-pos.py

For each query in the JSONL training file:
  - If the origin doc IS in the pos list  → keep only the origin doc in pos
    (remove all other pos entries), write the entry.
  - If the origin doc is NOT in the pos list → keep the entry unchanged.

The only difference from 4-29-exclude-other-pos.py is that entries where
the origin doc is absent from pos are retained as-is rather than discarded.

Inputs:
  - JSONL file:  e.g. dbpedia-gen-10k-v1-1entity-doc2query-cot.jsonl
  - JSON file(s): DiverseQuery_Main_*.json or DiverseQuery_FromFile_*.json
    (query → origin doc mapping)
Output:
  - Filtered JSONL file written next to the input file with suffix
    "_exclude_other_pos_save_non_pos.jsonl"

Example:
  python corpus_preprocess/4-29-exclude-other-pos-save-non-pos.py \
    --input-jsonl /data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/dbpedia-10k-0531-fill-easy-7.jsonl \
    --diverse-query /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/dbpedia_20260531_205905/DiverseQuery_Main_20260531_223659.json \
    --output /data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/dbpedia-10k-0531-fill-easy-7_exclude_other_pos_save_non_pos.jsonl
"""

import argparse
import json
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
DEFAULT_JSONL_PATH = Path(
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/ablation-5-self-refine/without-self-refine.jsonl"
)
DEFAULT_DIVERSE_QUERY_PATH = Path(
    "/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/msmarco_20260505_221851/DiverseQuery_Main_20260505_233813.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only the origin doc in pos when it appears in pos."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_JSONL_PATH,
        help="Input training-set JSONL path.",
    )
    parser.add_argument(
        "--diverse-query",
        type=Path,
        nargs="+",
        default=DEFAULT_DIVERSE_QUERY_PATH,
        help=(
            "One or more DiverseQuery_Main_*.json or DiverseQuery_FromFile_*.json "
            "paths used for query -> origin doc mapping."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Defaults to input stem plus "
            "'_exclude_other_pos_save_non_pos.jsonl'."
        ),
    )
    return parser.parse_args()


args = parse_args()
JSONL_PATH = args.input_jsonl
DIVERSE_QUERY_PATHS = (
    args.diverse_query
    if isinstance(args.diverse_query, list)
    else [args.diverse_query]
)
OUTPUT_PATH = args.output or (
    JSONL_PATH.parent / (JSONL_PATH.stem + "_exclude_other_pos_save_non_pos.jsonl")
)


if not JSONL_PATH.is_file():
    raise FileNotFoundError(f"Input JSONL not found: {JSONL_PATH}")
for path in DIVERSE_QUERY_PATHS:
    if not path.is_file():
        raise FileNotFoundError(f"Diverse-query file not found: {path}")


# ── step 1: build query → origin_doc mapping ──────────────────────────────────
query_to_origin_doc: dict[str, str] = {}
for diverse_query_path in DIVERSE_QUERY_PATHS:
    print(f"Loading diverse-query file: {diverse_query_path}")
    with open(diverse_query_path, "r", encoding="utf-8") as f:
        diverse_data = json.load(f)

    results = diverse_data.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"missing or invalid 'results' array: {diverse_query_path}")

    for result in results:
        origin_doc = result.get("doc", "")
        for q_obj in result.get("queries", []):
            query_text = q_obj.get("query", "")
            if query_text:
                query_to_origin_doc[query_text] = origin_doc

print(f"  Loaded {len(query_to_origin_doc):,} query → origin-doc mappings")


# ── step 2: process the JSONL file ────────────────────────────────────────────
print(f"\nProcessing JSONL file: {JSONL_PATH}")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

total = 0
no_mapping = 0          # query not found in diverse-query mapping (kept as-is)
ori_not_in_pos = 0      # origin doc not in pos (kept as-is)
other_pos_removed = 0   # entries where extra pos entries were stripped
ori_also_in_neg = 0     # entries where origin doc also appears in neg (stat only)

with (
    open(JSONL_PATH, "r", encoding="utf-8") as fin,
    open(OUTPUT_PATH, "w", encoding="utf-8") as fout,
):
    for line in fin:
        line = line.strip()
        if not line:
            continue
        total += 1
        entry = json.loads(line)

        query = entry.get("query", "")
        origin_doc = query_to_origin_doc.get(query)

        pos = entry.get("pos", [])
        neg = entry.get("neg", [])

        if not origin_doc:
            # No mapping — keep entry unchanged
            no_mapping += 1
        elif origin_doc not in pos:
            # Origin doc absent from pos — keep entry unchanged
            ori_not_in_pos += 1
        else:
            # Origin doc is in pos — strip all other pos entries
            if len(pos) > 1:
                other_pos_removed += 1
            entry["pos"] = [origin_doc]

            # Stats: is origin doc also in neg?
            if origin_doc in neg:
                ori_also_in_neg += 1

        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  Total entries read                  : {total:,}")
print(f"  Entries kept unchanged              : {no_mapping + ori_not_in_pos:,}")
print(f"    - query not in mapping            : {no_mapping:,}")
print(f"    - origin doc not in pos           : {ori_not_in_pos:,}")
print(f"  Entries with other pos removed      : {other_pos_removed:,}")
print(f"  Origin doc also in neg (stat only)  : {ori_also_in_neg:,}")
print(f"  Output written to                   : {OUTPUT_PATH}")
print(f"{'─'*60}")
