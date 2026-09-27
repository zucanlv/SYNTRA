#!/usr/bin/env python3
"""Align DeepSeek annotations with sampled items and compare judge agreement."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import ijson

try:
    from scripts.analyze_agreement import (
        BINARY_LABELS,
        THREE_LABELS,
        classification_metrics,
        normalize_classification,
        score_to_three_class,
        three_to_binary,
        write_csv,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/compare_deepseek.py
    from analyze_agreement import (  # type: ignore[no-redef]
        BINARY_LABELS,
        THREE_LABELS,
        classification_metrics,
        normalize_classification,
        score_to_three_class,
        three_to_binary,
        write_csv,
    )


DEFAULT_DEEPSEEK_PATHS = {
    "msmarco": Path(
        "/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/"
        "Main_Pipeline/Results/msmarco_20260718_214102/"
        "Annotated_Main_20260719_013023.json"
    ),
    "text2sql": Path(
        "/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/"
        "Main_Pipeline/Results/synthetic-text2sql_20260718_215629/"
        "Annotated_Main_20260718_231735.json"
    ),
    "theoremqa-theorems": Path(
        "/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/"
        "Main_Pipeline/Results/bright-documents-theoremqa_theorems_20260720_010413/"
        "Annotated_Main_20260720_022439.json"
    ),
}


@dataclass(frozen=True)
class Target:
    sample_id: str
    dataset: str
    query: str
    document: str
    source_doc_id: str


@dataclass(frozen=True)
class DeepSeekAnnotation:
    sample_id: str
    doc_id: str
    score: int
    classification: str
    reasoning: str
    match_method: str


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_targets(path: Path) -> tuple[list[str], dict[str, Target], dict[str, int]]:
    rows = read_csv_rows(path)
    order: list[str] = []
    targets: dict[str, Target] = {}
    qwen_scores: dict[str, int] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in targets:
            raise ValueError(f"{path}: duplicate sample_id {sample_id}")
        score = int(row["llm_score"])
        expected = score_to_three_class(score)
        observed = normalize_classification(row["llm_classification"])
        if observed != expected:
            raise ValueError(
                f"{path}: Qwen score/classification mismatch for {sample_id}: "
                f"{score!r} vs {row['llm_classification']!r}"
            )
        order.append(sample_id)
        targets[sample_id] = Target(
            sample_id=sample_id,
            dataset=row["dataset"],
            query=row["query"],
            document=row["document"],
            source_doc_id=str(row["source_doc_id"]),
        )
        qwen_scores[sample_id] = score
    return order, targets, qwen_scores


def load_human_consensus(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in result:
            raise ValueError(f"{path}: duplicate sample_id {sample_id}")
        if row["human_three_class"] not in THREE_LABELS:
            raise ValueError(f"{path}: invalid human label for {sample_id}")
        result[sample_id] = row
    return result


def _candidate_priority(candidate: dict[str, object], target: Target) -> tuple[int, str] | None:
    doc_id_match = str(candidate.get("doc_id")) == target.source_doc_id
    document_match = candidate.get("doc") == target.document
    if doc_id_match and document_match:
        return 0, "query_doc_id_document"
    if doc_id_match:
        return 1, "query_doc_id"
    if document_match:
        return 2, "query_document"
    return None


def extract_deepseek_annotations(
    path: Path, targets: Sequence[Target], require_complete: bool = True
) -> tuple[dict[str, DeepSeekAnnotation], dict[str, object]]:
    """Stream one Annotated_Main file and resolve each requested query-document pair."""
    by_query: dict[str, list[Target]] = defaultdict(list)
    for target in targets:
        by_query[target.query].append(target)

    raw_matches: dict[str, list[tuple[int, DeepSeekAnnotation]]] = defaultdict(list)
    query_occurrences: Counter[str] = Counter()
    with path.open("rb") as handle:
        for query_obj in ijson.items(handle, "results.item.queries.item"):
            query = query_obj.get("query")
            if query not in by_query:
                continue
            query_occurrences[str(query)] += 1
            annotations_by_doc_id: dict[str, list[dict[str, object]]] = defaultdict(list)
            for annotation in query_obj.get("annotations", []):
                annotations_by_doc_id[str(annotation.get("doc_id"))].append(annotation)

            for target in by_query[str(query)]:
                for candidate in query_obj.get("candidates", []):
                    priority_and_method = _candidate_priority(candidate, target)
                    if priority_and_method is None:
                        continue
                    priority, method = priority_and_method
                    doc_id = str(candidate.get("doc_id"))
                    for annotation in annotations_by_doc_id.get(doc_id, []):
                        score = int(annotation["score"])
                        expected = score_to_three_class(score)
                        observed = normalize_classification(str(annotation["classification"]))
                        if observed != expected:
                            raise ValueError(
                                f"{path}: DeepSeek score/classification mismatch for "
                                f"{target.sample_id}: {score!r} vs "
                                f"{annotation['classification']!r}"
                            )
                        raw_matches[target.sample_id].append(
                            (
                                priority,
                                DeepSeekAnnotation(
                                    sample_id=target.sample_id,
                                    doc_id=doc_id,
                                    score=score,
                                    classification=expected,
                                    reasoning=str(annotation.get("reasoning", "")),
                                    match_method=method,
                                ),
                            )
                        )

    resolved: dict[str, DeepSeekAnnotation] = {}
    duplicate_match_counts: dict[str, int] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    for target in targets:
        matches = raw_matches.get(target.sample_id, [])
        if not matches:
            missing.append(target.sample_id)
            continue
        best_priority = min(priority for priority, _ in matches)
        best = [item for priority, item in matches if priority == best_priority]
        unique = {
            (
                item.doc_id,
                item.score,
                item.classification,
                item.reasoning,
                item.match_method,
            ): item
            for item in best
        }
        if len(unique) != 1:
            ambiguous.append(target.sample_id)
            continue
        resolved[target.sample_id] = next(iter(unique.values()))
        if len(best) > 1:
            duplicate_match_counts[target.sample_id] = len(best)

    if ambiguous or (missing and require_complete):
        raise ValueError(
            f"{path}: alignment failed; missing={missing}, ambiguous={ambiguous}"
        )
    diagnostics: dict[str, object] = {
        "target_count": len(targets),
        "aligned_count": len(resolved),
        "match_methods": dict(Counter(item.match_method for item in resolved.values())),
        "queries_found": len(query_occurrences),
        "query_occurrence_count": sum(query_occurrences.values()),
        "duplicate_identical_match_count": len(duplicate_match_counts),
        "duplicate_identical_matches": duplicate_match_counts,
        "missing_count": len(missing),
        "missing_sample_ids": missing,
    }
    return resolved, diagnostics


def verify_annotation_model(annotated_path: Path) -> tuple[Path, str]:
    configs = sorted(annotated_path.parent.glob("config*.yaml"))
    if len(configs) != 1:
        raise ValueError(
            f"{annotated_path.parent}: expected one config YAML, found {len(configs)}"
        )
    config_path = configs[0]
    text = config_path.read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^llm_stages:\s*$.*?^  annotation:\s*$\s*^    model:\s*[\"']?([^\"'\s#]+)",
        text,
    )
    if not match:
        raise ValueError(f"{config_path}: cannot find llm_stages.annotation.model")
    model = match.group(1)
    if model != "deepseek-v4-flash":
        raise ValueError(f"{config_path}: annotation model is {model!r}, not deepseek-v4-flash")
    if not re.search(r"(?m)^fully_annotation:\s*true\s*(?:#.*)?$", text):
        raise ValueError(f"{config_path}: fully_annotation is not true")
    return config_path, model


def named_metrics(
    left: Sequence[str],
    right: Sequence[str],
    labels: Sequence[str],
    left_name: str,
    right_name: str,
) -> dict[str, object]:
    metrics = classification_metrics(left, right, labels)
    metrics["left_distribution"] = metrics.pop("human_distribution")
    metrics["right_distribution"] = metrics.pop("llm_distribution")
    metrics["confusion_matrix_rows"] = left_name
    metrics["confusion_matrix_columns"] = right_name
    return metrics


def analyze_pair(
    rows: Sequence[dict[str, object]], left_prefix: str, right_prefix: str
) -> dict[str, object]:
    return {
        "three_class": named_metrics(
            [str(row[f"{left_prefix}_three_class"]) for row in rows],
            [str(row[f"{right_prefix}_three_class"]) for row in rows],
            THREE_LABELS,
            left_prefix,
            right_prefix,
        ),
        "binary": named_metrics(
            [str(row[f"{left_prefix}_binary"]) for row in rows],
            [str(row[f"{right_prefix}_binary"]) for row in rows],
            BINARY_LABELS,
            left_prefix,
            right_prefix,
        ),
    }


def validate_same_ids(expected: Iterable[str], actual: Iterable[str], name: str) -> None:
    expected_set = set(expected)
    actual_set = set(actual)
    if expected_set != actual_set:
        raise ValueError(
            f"{name}: missing={sorted(expected_set - actual_set)}, "
            f"extra={sorted(actual_set - expected_set)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare human and Qwen3-30B labels with DeepSeek-V4-Flash."
    )
    parser.add_argument("--qwen", type=Path, default=Path("private/answer_key.csv"))
    parser.add_argument(
        "--human-consensus", type=Path, default=Path("results/human_consensus.csv")
    )
    parser.add_argument(
        "--deepseek-msmarco", type=Path, default=DEFAULT_DEEPSEEK_PATHS["msmarco"]
    )
    parser.add_argument(
        "--deepseek-text2sql", type=Path, default=DEFAULT_DEEPSEEK_PATHS["text2sql"]
    )
    parser.add_argument(
        "--deepseek-theoremqa",
        type=Path,
        default=DEFAULT_DEEPSEEK_PATHS["theoremqa-theorems"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deepseek_paths = {
        "msmarco": args.deepseek_msmarco,
        "text2sql": args.deepseek_text2sql,
        "theoremqa-theorems": args.deepseek_theoremqa,
    }
    order, targets, qwen_scores = load_targets(args.qwen)
    human = load_human_consensus(args.human_consensus)
    validate_same_ids(order, human, str(args.human_consensus))

    deepseek: dict[str, DeepSeekAnnotation] = {}
    source_metadata: dict[str, object] = {}
    for dataset, path in deepseek_paths.items():
        config_path, model = verify_annotation_model(path)
        dataset_targets = [target for target in targets.values() if target.dataset == dataset]
        extracted, diagnostics = extract_deepseek_annotations(
            path, dataset_targets, require_complete=False
        )
        deepseek.update(extracted)
        source_metadata[dataset] = {
            "annotated_path": str(path),
            "config_path": str(config_path),
            "annotation_model": model,
            "fully_annotation": True,
            "alignment": diagnostics,
        }

    records: list[dict[str, object]] = []
    for sample_id in order:
        if sample_id not in deepseek:
            continue
        target = targets[sample_id]
        human_row = human[sample_id]
        human_three = human_row["human_three_class"]
        human_binary = human_row["human_binary"]
        qwen_score = qwen_scores[sample_id]
        qwen_three = score_to_three_class(qwen_score)
        qwen_binary = three_to_binary(qwen_three)
        deepseek_item = deepseek[sample_id]
        deepseek_three = deepseek_item.classification
        deepseek_binary = three_to_binary(deepseek_three)
        records.append(
            {
                "sample_id": sample_id,
                "dataset": target.dataset,
                "human_three_class": human_three,
                "qwen_score": qwen_score,
                "qwen_three_class": qwen_three,
                "deepseek_score": deepseek_item.score,
                "deepseek_three_class": deepseek_three,
                "human_binary": human_binary,
                "qwen_binary": qwen_binary,
                "deepseek_binary": deepseek_binary,
                "human_deepseek_three_match": human_three == deepseek_three,
                "human_deepseek_binary_match": human_binary == deepseek_binary,
                "qwen_deepseek_three_match": qwen_three == deepseek_three,
                "qwen_deepseek_binary_match": qwen_binary == deepseek_binary,
                "needs_adjudication": human_row["needs_adjudication"],
                "deepseek_doc_id": deepseek_item.doc_id,
                "deepseek_match_method": deepseek_item.match_method,
            }
        )

    datasets = list(dict.fromkeys(targets[sample_id].dataset for sample_id in order))
    majority_only = [row for row in records if row["needs_adjudication"] != "True"]
    coverage_by_dataset = {
        dataset: {
            "requested": sum(target.dataset == dataset for target in targets.values()),
            "strictly_aligned": sum(row["dataset"] == dataset for row in records),
        }
        for dataset in datasets
    }
    report = {
        "mapping": {
            "positive": [2, 3],
            "hard_negative": [1],
            "easy_negative": [0],
            "binary_negative": ["hard_negative", "easy_negative"],
        },
        "requested_sample_count": len(order),
        "strictly_aligned_sample_count": len(records),
        "strict_alignment_definition": "exact query plus matching candidate document ID/text",
        "comparability": {
            "full_requested_comparison_valid": len(records) == len(order),
            "strict_alignment_coverage": len(records) / len(order),
            "coverage_by_dataset": coverage_by_dataset,
            "metric_scope": "strict overlap only",
            "warning": (
                "The Qwen and DeepSeek experiments regenerated queries independently. "
                "Metrics below do not estimate full-dataset judge agreement when strict "
                "alignment coverage is incomplete."
            ),
        },
        "source_metadata": source_metadata,
        "human_vs_deepseek": {
            "all_samples": analyze_pair(records, "human", "deepseek"),
            "majority_only_excluding_three_way_ties": analyze_pair(
                majority_only, "human", "deepseek"
            ),
            "by_dataset": {
                dataset: analyze_pair(
                    [row for row in records if row["dataset"] == dataset],
                    "human",
                    "deepseek",
                )
                for dataset in datasets
            },
        },
        "qwen3_30b_vs_deepseek": {
            "all_samples": analyze_pair(records, "qwen", "deepseek"),
            "by_dataset": {
                dataset: analyze_pair(
                    [row for row in records if row["dataset"] == dataset],
                    "qwen",
                    "deepseek",
                )
                for dataset in datasets
            },
        },
    }

    fields = [
        "sample_id",
        "dataset",
        "human_three_class",
        "qwen_score",
        "qwen_three_class",
        "deepseek_score",
        "deepseek_three_class",
        "human_binary",
        "qwen_binary",
        "deepseek_binary",
        "human_deepseek_three_match",
        "human_deepseek_binary_match",
        "qwen_deepseek_three_match",
        "qwen_deepseek_binary_match",
        "needs_adjudication",
        "deepseek_doc_id",
        "deepseek_match_method",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "deepseek_aligned_annotations.csv", records, fields)
    write_csv(
        args.output_dir / "deepseek_disagreements.csv",
        [
            row
            for row in records
            if not row["human_deepseek_three_match"]
            or not row["qwen_deepseek_three_match"]
        ],
        fields,
    )
    report_path = args.output_dir / "deepseek_comparison_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "requested_sample_count": len(order),
                "strictly_aligned_sample_count": len(records),
                "human_vs_deepseek": report["human_vs_deepseek"]["all_samples"],
                "qwen3_30b_vs_deepseek": report["qwen3_30b_vs_deepseek"]["all_samples"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
