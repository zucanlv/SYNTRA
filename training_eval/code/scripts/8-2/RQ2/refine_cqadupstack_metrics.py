#!/usr/bin/env python3
"""Add per-subtask CQADupStack metrics to existing BEIR evaluation outputs.

This script only consumes saved search-result JSON files. It does not load a
retrieval model, generate embeddings, or run retrieval again.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
FLAGEMBEDDING_ROOT = PROJECT_ROOT / "third_party" / "FlagEmbedding"
if str(FLAGEMBEDDING_ROOT) not in sys.path:
    sys.path.insert(0, str(FLAGEMBEDDING_ROOT))

from FlagEmbedding.abc.evaluation.utils import evaluate_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="An evaluation directory or a directory recursively containing beir_eval_results.md files.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "beir" / "eval" / "data",
        help="BEIR dataset directory containing cqadupstack/<subtask>/test_qrels.jsonl.",
    )
    parser.add_argument(
        "--update-markdown",
        action="store_true",
        help="Replace/add the 'CQADupStack subtask results' section in each report.",
    )
    return parser.parse_args()


def find_reports(eval_root: Path) -> List[Path]:
    if eval_root.is_file():
        if eval_root.name != "beir_eval_results.md":
            raise ValueError("--eval-root file must be named beir_eval_results.md")
        return [eval_root]
    reports = sorted(eval_root.rglob("beir_eval_results.md"))
    if not reports:
        raise FileNotFoundError(f"No beir_eval_results.md found under {eval_root}")
    return reports


def load_qrels(dataset_dir: Path, subtask: str) -> Dict[str, Dict[str, int]]:
    qrels_path = dataset_dir / "cqadupstack" / subtask / "test_qrels.jsonl"
    if not qrels_path.is_file():
        raise FileNotFoundError(f"Missing qrels for subtask '{subtask}': {qrels_path}")

    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with qrels_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            record = json.loads(line)
            try:
                qrels[str(record["qid"])][str(record["docid"])] = int(record["relevance"])
            except KeyError as error:
                raise ValueError(f"Invalid qrels record at {qrels_path}:{line_number}") from error
    return dict(qrels)


def iter_cqadupstack_results(report_path: Path) -> Iterable[Tuple[Path, dict]]:
    search_root = report_path.parent / "search_results"
    if not search_root.is_dir():
        return
    for result_path in sorted(search_root.rglob("cqadupstack-*-test.json")):
        with result_path.open(encoding="utf-8") as file:
            record = json.load(file)
        if record.get("dataset_name") == "cqadupstack" and record.get("sub_dataset_name"):
            yield result_path, record


def compute_rows(report_path: Path, dataset_dir: Path) -> List[Tuple[str, str, str, float, float]]:
    rows = []
    for result_path, record in iter_cqadupstack_results(report_path):
        subtask = str(record["sub_dataset_name"])
        qrels = load_qrels(dataset_dir, subtask)
        search_results = record.get("search_results")
        if not isinstance(search_results, dict):
            raise ValueError(f"Missing search_results in {result_path}")

        missing_queries = set(qrels) - set(search_results)
        if missing_queries:
            raise ValueError(
                f"{result_path} misses {len(missing_queries)} qrels queries; refusing to compute partial metrics."
            )

        ndcg, _, recall, _ = evaluate_metrics(
            qrels=qrels,
            results=search_results,
            k_values=[10, 100],
            scores_basename=f"cqadupstack-{subtask}-test",
        )
        rows.append(
            (
                str(record.get("model_name", "unknown")),
                str(record.get("reranker_name", "unknown")),
                subtask,
                ndcg["NDCG@10"] * 100,
                recall["Recall@100"] * 100,
            )
        )
    return sorted(rows, key=lambda row: (row[0], row[1], row[2]))


def render_section(rows: List[Tuple[str, str, str, float, float]]) -> str:
    lines = ["## CQADupStack subtask results", ""]
    grouped: Dict[Tuple[str, str], List[Tuple[str, float, float]]] = defaultdict(list)
    for model, reranker, subtask, ndcg, recall in rows:
        grouped[(model, reranker)].append((subtask, ndcg, recall))

    for index, ((model, reranker), group_rows) in enumerate(sorted(grouped.items())):
        if len(grouped) > 1:
            lines.extend([f"### Model: {model}; Reranker: {reranker}", ""])
        lines.extend([
            "| Subtask | ndcg_at_10 | recall_at_100 |",
            "| :--- | ---: | ---: |",
        ])
        lines.extend(
            f"| {subtask} | **{ndcg:.3f}** | **{recall:.3f}** |"
            for subtask, ndcg, recall in group_rows
        )
        if index != len(grouped) - 1:
            lines.append("")
    return "\n".join(lines) + "\n"


def update_report(report_path: Path, section: str) -> None:
    text = report_path.read_text(encoding="utf-8")
    pattern = r"(?ms)^## CQADupStack subtask results\n.*?(?=^## |\Z)"
    if re.search(pattern, text):
        updated = re.sub(pattern, section.rstrip(), text).rstrip() + "\n"
    else:
        updated = text.rstrip() + "\n\n" + section
    report_path.write_text(updated, encoding="utf-8")


def main() -> None:
    args = parse_args()
    reports = find_reports(args.eval_root)
    processed = 0
    for report_path in reports:
        rows = compute_rows(report_path, args.dataset_dir)
        if not rows:
            continue
        section = render_section(rows)
        print(f"\n# {report_path}\n\n{section}", end="")
        if args.update_markdown:
            update_report(report_path, section)
        processed += 1

    if processed == 0:
        raise SystemExit("No CQADupStack search-result JSON files found.")
    action = "updated" if args.update_markdown else "reported"
    print(f"\n{action.capitalize()} {processed} CQADupStack evaluation report(s).")


if __name__ == "__main__":
    main()
