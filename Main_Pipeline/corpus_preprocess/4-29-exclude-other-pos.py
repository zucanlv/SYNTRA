"""
Script: 4-29-exclude-other-pos.py

For each query in the JSONL training file:
  - If the origin doc IS in the pos list  → keep only the origin doc in pos
    (remove all other pos entries), remove origin doc from neg if present.
  - If the origin doc is NOT in the pos list → discard the entry entirely.

Inputs:
  - JSONL file:  e.g. dbpedia-gen-10k-v1-1entity-doc2query-cot.jsonl
  - JSON file:   DiverseQuery_Main_*.json  (query → origin doc mapping)
Output:
  - Filtered JSONL file written next to the input file with suffix
    "_exclude_other_pos.jsonl"
"""

import json
import os
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
JSONL_PATH = Path(
    "/data/share/project/shared_datasets/DSA/syn_data/msmarco/title-msmarco-10k/title-msmarco-74k-gen-v2-10k-1entity-doc2query_cot.jsonl"
)
DIVERSE_QUERY_PATH = Path(
    "/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/msmarco_20260427_052122/DiverseQuery_Main_20260427_064237.json"
)
OUTPUT_PATH = JSONL_PATH.parent / (JSONL_PATH.stem + "_exclude_other_pos.jsonl")


# ── step 1: build query → origin_doc mapping ──────────────────────────────────
print(f"Loading diverse-query file: {DIVERSE_QUERY_PATH}")
with open(DIVERSE_QUERY_PATH, "r", encoding="utf-8") as f:
    diverse_data = json.load(f)

query_to_origin_doc: dict[str, str] = {}
for result in diverse_data.get("results", []):
    origin_doc = result.get("doc", "")
    for q_obj in result.get("queries", []):
        query_text = q_obj.get("query", "")
        if query_text:
            query_to_origin_doc[query_text] = origin_doc

print(f"  Loaded {len(query_to_origin_doc):,} query → origin-doc mappings")


# ── step 2: filter the JSONL file ─────────────────────────────────────────────
print(f"\nProcessing JSONL file: {JSONL_PATH}")

total = 0
kept = 0
discarded_no_origin = 0   # query not found in diverse-query mapping
discarded_ori_not_in_pos = 0  # origin doc found but not present in pos list
other_pos_removed = 0     # entries where extra pos entries were removed
ori_removed_from_neg = 0  # entries where origin doc was removed from neg

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

        # No mapping found for this query — skip
        if not origin_doc:
            discarded_no_origin += 1
            continue

        # Origin doc not in pos — discard
        if origin_doc not in pos:
            discarded_ori_not_in_pos += 1
            continue

        # Keep only origin doc in pos
        if len(pos) > 1:
            other_pos_removed += 1
        new_pos = [origin_doc]

        # Count entries where origin doc also appears in neg (for stats only)
        if origin_doc in neg:
            ori_removed_from_neg += 1

        entry["pos"] = new_pos
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        kept += 1

# ── summary ───────────────────────────────────────────────────────────────────
discarded = total - kept
print(f"\n{'─'*55}")
print(f"  Total entries read            : {total:,}")
print(f"  Entries kept                  : {kept:,}")
print(f"  Entries discarded             : {discarded:,}")
print(f"    - query not in mapping      : {discarded_no_origin:,}")
print(f"    - origin doc not in pos     : {discarded_ori_not_in_pos:,}")
print(f"  Other pos entries removed     : {other_pos_removed:,}")
print(f"  Origin doc also in neg        : {ori_removed_from_neg:,}  (kept, not removed)")
print(f"  Output written to             : {OUTPUT_PATH}")
print(f"{'─'*55}")
