#!/usr/bin/env python3
"""从 Annotated_DiverseQuery 结果中提取 query / positive passages / negative passages，输出简洁 JSON。"""

import json
import sys
from pathlib import Path

# 默认输入输出路径
INPUT_JSON = Path(__file__).resolve().parent.parent / "Results" / "Annotated_DiverseQuery_Batch_20260203_043528_592874_one_1.json"
OUTPUT_JSON = Path(__file__).resolve().parent.parent / "Results" / "triplets_view.json"


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_JSON
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_JSON

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    triplets = []
    for item in data.get("results", []):
        for q in item.get("queries", []):
            pid_to_passage = {c["pid"]: c.get("text", c.get("passage", "")) for c in q.get("candidates", [])}
            pos_pids = q.get("positive_pids", [])
            hard_neg_pids = q.get("hard_negative_pids", [])
            easy_neg_pids = q.get("easy_negative_pids", [])
            
            triplets.append({
                "query": q["query"],
                "positives": [pid_to_passage[pid] for pid in pos_pids if pid in pid_to_passage],
                "hard_negatives": [pid_to_passage[pid] for pid in hard_neg_pids if pid in pid_to_passage],
                "easy_negatives": [pid_to_passage[pid] for pid in easy_neg_pids if pid in pid_to_passage],
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)

    print(f"已提取 {len(triplets)} 条 triplet，写入: {output_path}")


if __name__ == "__main__":
    main()
