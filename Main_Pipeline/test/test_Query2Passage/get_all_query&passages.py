#!/usr/bin/env python3
"""
从 Query2Passage 的 JSON 结果中提取所有 query 及其对应的 passages，
便于分析观察数据。
"""

import json
from pathlib import Path


def load_json(path: str) -> dict:
    """加载 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_query_passages(data: dict) -> list[dict]:
    """
    从原始数据中提取 (query, original_passage, candidate_passages) 列表。
    每个元素包含：
      - query: 查询文本
      - original_passage: 生成该 query 的原文
      - candidate_passages: 检索得到的候选 passage 列表（含 pid, passage, score, rank, is_original）
    """
    extracted = []
    for item in data.get("results", []):
        original_passage = item.get("passage", "")
        for q in item.get("queries", []):
            query_text = q.get("query", "")
            candidates = [
                {
                    "pid": c.get("pid"),
                    "passage": c.get("passage", ""),
                    "score": c.get("score"),
                    "rank": c.get("rank"),
                    "is_original": c.get("is_original", False),
                }
                for c in q.get("candidates", [])
            ]
            extracted.append({
                "query": query_text,
                "original_passage": original_passage,
                "candidate_passages": candidates,
            })
    return extracted


def save_extracted_json(extracted: list[dict], out_path: str) -> None:
    """保存为 JSON，便于程序化分析。"""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)
    print(f"已保存 JSON: {out_path} (共 {len(extracted)} 条 query-passage 对)")


def save_readable_text(extracted: list[dict], out_path: str) -> None:
    """保存为可读文本，便于人工浏览。"""
    lines = []
    for i, item in enumerate(extracted, 1):
        lines.append("=" * 80)
        lines.append(f"【{i}】 QUERY")
        lines.append("-" * 40)
        lines.append(item["query"])
        lines.append("")
        lines.append("ORIGINAL PASSAGE (生成该 query 的原文)")
        lines.append("-" * 40)
        lines.append(item["original_passage"])
        lines.append("")
        lines.append("CANDIDATE PASSAGES (检索得到的候选)")
        lines.append("-" * 40)
        for j, c in enumerate(item["candidate_passages"], 1):
            orig_tag = " [ORIGINAL]" if c.get("is_original") else ""
            score = c.get("score")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            lines.append(f"  [{j}] rank={c.get('rank')} score={score_str}{orig_tag} pid={c.get('pid')}")
            lines.append(f"      {c['passage'][:200]}{'...' if len(c['passage']) > 200 else ''}")
            lines.append("")
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已保存可读文本: {out_path}")


def main():
    # 路径：相对于脚本所在目录的上一级 Main_Pipeline
    script_dir = Path(__file__).resolve().parent
    pipeline_dir = script_dir.parent
    input_path = pipeline_dir / "Results" / "Query2Passage_DiverseQuery_Batch_20260203_043528_592874_one.json"
    output_dir = script_dir

    if not input_path.exists():
        print(f"输入文件不存在: {input_path}")
        return

    data = load_json(str(input_path))
    extracted = extract_query_passages(data)

    # 输出文件
    out_json = output_dir / "extracted_queries_and_passages.json"
    out_txt = output_dir / "extracted_queries_and_passages.txt"

    save_extracted_json(extracted, str(out_json))
    save_readable_text(extracted, str(out_txt))

    # 简单统计
    total_queries = len(extracted)
    total_candidates = sum(len(e["candidate_passages"]) for e in extracted)
    print(f"\n统计: 共 {total_queries} 条 query, 共 {total_candidates} 条 candidate passage 记录。")


if __name__ == "__main__":
    main()
