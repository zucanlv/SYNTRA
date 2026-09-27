#!/usr/bin/env python3
"""Build human consensus labels and compare them with LLM annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


THREE_LABELS = ("positive", "hard_negative", "easy_negative")
BINARY_LABELS = ("positive", "negative")


def score_to_three_class(score: int) -> str:
    if score in {2, 3}:
        return "positive"
    if score == 1:
        return "hard_negative"
    if score == 0:
        return "easy_negative"
    raise ValueError(f"score must be 0, 1, 2, or 3, got {score!r}")


def three_to_binary(label: str) -> str:
    if label == "positive":
        return "positive"
    if label in {"hard_negative", "easy_negative"}:
        return "negative"
    raise ValueError(f"unknown three-class label: {label!r}")


def normalize_classification(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def read_samples(path: Path) -> tuple[list[str], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    order: list[str] = []
    datasets: dict[str, str] = {}
    for item in payload:
        sample_id = item["sample_id"]
        if sample_id in datasets:
            raise ValueError(f"duplicate sample_id in samples: {sample_id}")
        order.append(sample_id)
        datasets[sample_id] = item["dataset"]
    return order, datasets


def read_human_scores(path: Path, datasets: dict[str, str]) -> tuple[str, dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scores: dict[str, int] = {}
    annotator_ids: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in datasets:
            raise ValueError(f"{path}: unknown sample_id {sample_id}")
        if sample_id in scores:
            raise ValueError(f"{path}: duplicate sample_id {sample_id}")
        if row["dataset"] != datasets[sample_id]:
            raise ValueError(f"{path}: wrong dataset for {sample_id}")
        score = int(row["score"])
        score_to_three_class(score)
        scores[sample_id] = score
        annotator_ids.add(row["annotator_id"])
    if len(annotator_ids) != 1:
        raise ValueError(f"{path}: expected exactly one annotator_id")
    return next(iter(annotator_ids)), scores


def read_llm_scores(path: Path, datasets: dict[str, str]) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scores: dict[str, int] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in datasets:
            raise ValueError(f"{path}: unknown sample_id {sample_id}")
        if sample_id in scores:
            raise ValueError(f"{path}: duplicate sample_id {sample_id}")
        if row["dataset"] != datasets[sample_id]:
            raise ValueError(f"{path}: wrong dataset for {sample_id}")
        score = int(row["llm_score"])
        expected = score_to_three_class(score)
        observed = normalize_classification(row["llm_classification"])
        if observed != expected:
            raise ValueError(
                f"{path}: llm_score/classification mismatch for {sample_id}: "
                f"{score!r} vs {row['llm_classification']!r}"
            )
        scores[sample_id] = score
    return scores


def require_same_ids(expected: set[str], named_scores: Iterable[tuple[str, dict[str, int]]]) -> None:
    for name, scores in named_scores:
        actual = set(scores)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"{name}: missing={missing}, extra={extra}")


def consensus(votes: Sequence[str]) -> tuple[str, int, str, bool]:
    counts = Counter(votes)
    label, count = counts.most_common(1)[0]
    if count >= 2:
        return label, count, "majority", False
    if set(votes) == set(THREE_LABELS):
        return "hard_negative", 1, "ordinal_median_tiebreak", True
    raise ValueError(f"unexpected vote pattern: {votes!r}")


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(
    human: Sequence[str], llm: Sequence[str], labels: Sequence[str]
) -> dict[str, object]:
    if len(human) != len(llm):
        raise ValueError("human and LLM labels must have the same length")
    matrix = {
        true_label: {pred_label: 0 for pred_label in labels}
        for true_label in labels
    }
    for true_label, pred_label in zip(human, llm):
        matrix[true_label][pred_label] += 1

    n = len(human)
    matches = sum(matrix[label][label] for label in labels)
    observed = safe_div(matches, n)
    human_counts = Counter(human)
    llm_counts = Counter(llm)
    expected = safe_div(
        sum(human_counts[label] * llm_counts[label] for label in labels), n * n
    )
    kappa = (
        1.0
        if expected == 1.0 and observed == 1.0
        else safe_div(observed - expected, 1.0 - expected)
    )

    per_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": human_counts[label],
        }

    macro_precision = sum(item["precision"] for item in per_label.values()) / len(labels)
    macro_recall = sum(item["recall"] for item in per_label.values()) / len(labels)
    macro_f1 = sum(item["f1"] for item in per_label.values()) / len(labels)
    weighted_f1 = safe_div(
        sum(item["f1"] * item["support"] for item in per_label.values()), n
    )
    return {
        "n": n,
        "matches": matches,
        "agreement": observed,
        "cohen_kappa": kappa,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "human_distribution": {label: human_counts[label] for label in labels},
        "llm_distribution": {label: llm_counts[label] for label in labels},
        "per_label": per_label,
        "confusion_matrix": matrix,
    }


def fleiss_kappa(rows: Sequence[Sequence[str]], labels: Sequence[str]) -> float:
    if not rows:
        return 0.0
    raters = len(rows[0])
    if raters < 2 or any(len(row) != raters for row in rows):
        raise ValueError("Fleiss kappa requires a fixed number of at least two raters")
    category_totals = Counter()
    item_agreements: list[float] = []
    for row in rows:
        counts = Counter(row)
        category_totals.update(counts)
        item_agreements.append(
            safe_div(sum(count * count for count in counts.values()) - raters, raters * (raters - 1))
        )
    observed = sum(item_agreements) / len(item_agreements)
    total_ratings = len(rows) * raters
    expected = sum(
        (category_totals[label] / total_ratings) ** 2 for label in labels
    )
    if expected == 1.0 and observed == 1.0:
        return 1.0
    return safe_div(observed - expected, 1.0 - expected)


def analyze_scope(records: Sequence[dict[str, object]]) -> dict[str, object]:
    human_three = [str(row["human_three_class"]) for row in records]
    llm_three = [str(row["llm_three_class"]) for row in records]
    human_binary = [str(row["human_binary"]) for row in records]
    llm_binary = [str(row["llm_binary"]) for row in records]
    return {
        "three_class": classification_metrics(human_three, llm_three, THREE_LABELS),
        "binary": classification_metrics(human_binary, llm_binary, BINARY_LABELS),
    }


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vote three human annotations and compare consensus with LLM labels."
    )
    parser.add_argument("--samples", type=Path, default=Path("data/samples.json"))
    parser.add_argument(
        "--human",
        type=Path,
        nargs=3,
        default=[
            Path("annotations/annotator_1.csv"),
            Path("annotations/annotator_2.csv"),
            Path("annotations/annotation3.csv"),
        ],
    )
    parser.add_argument("--llm", type=Path, default=Path("private/answer_key.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_order, datasets = read_samples(args.samples)
    humans = [read_human_scores(path, datasets) for path in args.human]
    llm_scores = read_llm_scores(args.llm, datasets)
    expected_ids = set(sample_order)
    require_same_ids(
        expected_ids,
        [(str(path), scores) for path, (_, scores) in zip(args.human, humans)]
        + [(str(args.llm), llm_scores)],
    )

    records: list[dict[str, object]] = []
    for sample_id in sample_order:
        raw_scores = [scores[sample_id] for _, scores in humans]
        human_votes = [score_to_three_class(score) for score in raw_scores]
        human_label, agreement_count, method, needs_adjudication = consensus(human_votes)
        binary_votes = [three_to_binary(label) for label in human_votes]
        binary_label, binary_count, _, _ = consensus(binary_votes)
        llm_score = llm_scores[sample_id]
        llm_label = score_to_three_class(llm_score)
        llm_binary = three_to_binary(llm_label)
        records.append(
            {
                "sample_id": sample_id,
                "dataset": datasets[sample_id],
                "annotator_1_score": raw_scores[0],
                "annotator_2_score": raw_scores[1],
                "annotator_3_score": raw_scores[2],
                "annotator_1_label": human_votes[0],
                "annotator_2_label": human_votes[1],
                "annotator_3_label": human_votes[2],
                "human_three_class": human_label,
                "human_binary": binary_label,
                "agreement_count": agreement_count,
                "binary_agreement_count": binary_count,
                "consensus_method": method,
                "unanimous": agreement_count == 3,
                "needs_adjudication": needs_adjudication,
                "llm_score": llm_score,
                "llm_three_class": llm_label,
                "llm_binary": llm_binary,
                "three_class_match": human_label == llm_label,
                "binary_match": binary_label == llm_binary,
            }
        )

    dataset_names = list(dict.fromkeys(datasets[sample_id] for sample_id in sample_order))
    majority_only = [row for row in records if not row["needs_adjudication"]]
    human_three_rows = [
        [str(row[f"annotator_{index}_label"]) for index in (1, 2, 3)]
        for row in records
    ]
    human_binary_rows = [
        [three_to_binary(label) for label in row]
        for row in human_three_rows
    ]
    report = {
        "mapping": {
            "positive": [2, 3],
            "hard_negative": [1],
            "easy_negative": [0],
            "binary_negative": ["hard_negative", "easy_negative"],
        },
        "consensus_rule": {
            "default": "three-class majority vote",
            "three_way_tie": "ordinal median = hard_negative; flagged for adjudication",
            "binary": "majority vote after collapsing both negative classes",
        },
        "sample_count": len(records),
        "unanimous_count": sum(bool(row["unanimous"]) for row in records),
        "three_way_tie_count": sum(bool(row["needs_adjudication"]) for row in records),
        "three_way_tie_ids": [
            str(row["sample_id"]) for row in records if row["needs_adjudication"]
        ],
        "human_inter_annotator": {
            "three_class_fleiss_kappa": fleiss_kappa(human_three_rows, THREE_LABELS),
            "binary_fleiss_kappa": fleiss_kappa(human_binary_rows, BINARY_LABELS),
        },
        "llm_vs_human": {
            "all_samples": analyze_scope(records),
            "majority_only_excluding_three_way_ties": analyze_scope(majority_only),
            "by_dataset": {
                dataset: analyze_scope(
                    [row for row in records if row["dataset"] == dataset]
                )
                for dataset in dataset_names
            },
        },
    }

    consensus_fields = [
        "sample_id", "dataset",
        "annotator_1_score", "annotator_2_score", "annotator_3_score",
        "annotator_1_label", "annotator_2_label", "annotator_3_label",
        "human_three_class", "human_binary", "agreement_count",
        "binary_agreement_count", "consensus_method", "unanimous",
        "needs_adjudication",
    ]
    disagreement_fields = [
        "sample_id", "dataset", "human_three_class", "llm_score",
        "llm_three_class", "human_binary", "llm_binary",
        "three_class_match", "binary_match", "needs_adjudication",
    ]
    write_csv(args.output_dir / "human_consensus.csv", records, consensus_fields)
    write_csv(
        args.output_dir / "llm_human_disagreements.csv",
        [row for row in records if not row["three_class_match"]],
        disagreement_fields,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "agreement_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = report["llm_vs_human"]["all_samples"]
    print(json.dumps({
        "sample_count": report["sample_count"],
        "unanimous_count": report["unanimous_count"],
        "three_way_tie_count": report["three_way_tie_count"],
        "three_class": summary["three_class"],
        "binary": summary["binary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
