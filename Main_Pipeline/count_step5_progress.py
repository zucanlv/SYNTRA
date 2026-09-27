#!/usr/bin/env python3
"""
从 main 流水线日志中统计 Step 5（BatchCandidateAnnotator）已处理进度。

日志里不会对每个 query 打一行完成标记；并发标注时 httpx 会为每次成功的
chat/completions 请求记一行 "HTTP/1.1 200 OK"。若配置了 candidates_num_each_anno
且小于候选数，则一个 query 会对应多次 HTTP 调用，需用 --top-k 与 --cna 换算。

用法:
  python count_step5_progress.py path/to/main_*.log
  python count_step5_progress.py path/to/main_*.log --top-k 20 --cna 4
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


STEP5_BEGIN = re.compile(r"=== Step 5 ===\s+task=")
STEP5_DONE = re.compile(r"=== Step 5 done ===")
STARTING_FILE = re.compile(r"Starting file processing:\s*(\d+)\s*queries")
HTTP_200 = re.compile(r'HTTP Request: POST .+ "HTTP/1\.1 200 OK"')
# Batch_Candidate_Annotator: "Annotation complete. ... Stats: {'total_queries': N, ...}"
STATS_TOTAL_QUERIES = re.compile(r"Annotation complete\..*Stats:.*['\"]total_queries['\"]:\s*(\d+)")


def calls_per_query(top_k: int | None, cna: int | None) -> int:
    if top_k is None and cna is None:
        return 1
    if top_k is None or cna is None:
        raise ValueError("必须同时提供 --top-k 与 --cna，或两者都不提供（按每 query 1 次 HTTP 计）")
    if cna <= 0:
        raise ValueError("--cna 必须为正整数")
    return max(1, math.ceil(top_k / cna))


def parse_log(
    path: Path,
    *,
    top_k: int | None,
    cna: int | None,
) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    step5_line = None
    for i, line in enumerate(lines):
        if STEP5_BEGIN.search(line):
            step5_line = i
            break
    if step5_line is None:
        raise SystemExit("未找到 '=== Step 5 ==='，请确认是 main 流水线日志且已跑到 Step 5。")

    total_planned: int | None = None
    for line in lines:
        m = STARTING_FILE.search(line)
        if m:
            total_planned = int(m.group(1))
            break

    http_in_step5 = 0
    for line in lines[step5_line:]:
        if STEP5_DONE.search(line):
            break
        if HTTP_200.search(line):
            http_in_step5 += 1

    cpp = calls_per_query(top_k, cna)
    queries_done = http_in_step5 // cpp
    remainder_http = http_in_step5 % cpp

    step5_finished = any(STEP5_DONE.search(line) for line in lines[step5_line:])

    total_queries_from_stats: int | None = None
    for line in lines[step5_line:]:
        m = STATS_TOTAL_QUERIES.search(line)
        if m:
            total_queries_from_stats = int(m.group(1))
            break

    return {
        "log_path": str(path),
        "step5_begins_at_line": step5_line + 1,
        "total_queries_planned": total_planned,
        "http_200_in_step5": http_in_step5,
        "llm_calls_assumed_per_query": cpp,
        "queries_completed_estimate": queries_done,
        "in_flight_http_calls": remainder_http,
        "step5_finished": step5_finished,
        "total_queries_from_stats": total_queries_from_stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="统计 main 日志中 Step 5 已处理 query 数量（基于 HTTP 200 次数）。"
    )
    ap.add_argument("log_path", type=Path, help="main_*.log 路径")
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="K",
        help="与 config 中 top_k_candidates 一致；与 --cna 一起用于计算每 query 的 LLM 次数",
    )
    ap.add_argument(
        "--cna",
        type=int,
        default=None,
        metavar="N",
        dest="candidates_num_each_anno",
        help="与 config 中 candidates_num_each_anno 一致",
    )
    args = ap.parse_args()

    if not args.log_path.is_file():
        print(f"文件不存在: {args.log_path}", file=sys.stderr)
        sys.exit(1)

    try:
        r = parse_log(
            args.log_path,
            top_k=args.top_k,
            cna=args.candidates_num_each_anno,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print(f"日志: {r['log_path']}")
    print(f"Step 5 起始行: {r['step5_begins_at_line']}")
    if r["total_queries_planned"] is not None:
        print(f"计划处理 query 数: {r['total_queries_planned']}")
    print(f"Step 5 内 HTTP 200 次数: {r['http_200_in_step5']}")
    print(f"假定每 query LLM 调用数: {r['llm_calls_assumed_per_query']}")
    if r["total_queries_from_stats"] is not None:
        print(f"日志中 Annotation 统计的 total_queries（精确）: {r['total_queries_from_stats']}")
    print(f"由 HTTP 次数推算的已处理 query 数: {r['queries_completed_estimate']}")
    if r["in_flight_http_calls"]:
        print(f"余数 HTTP（可能仍在进行当前 query 的后续 chunk）: {r['in_flight_http_calls']}")
    print(f"Step 5 是否已结束: {r['step5_finished']}")


if __name__ == "__main__":
    main()
