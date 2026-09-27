#!/usr/bin/env python3
"""Stream EvidenceBench-100k into sentence corpora and train-only few-shots.

The full test corpus preserves every sentence and its original order while
deliberately excluding hypotheses and labels. A filtered subset is emitted for
query generation. Few-shot query-positive pairs come only from labeled train
records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

import ijson


DEFAULT_DATASET_DIR = Path(
    "/data/share/project/shared_datasets/DSA/datasets/EvidenceBench-100k"
)
DEFAULT_TRAIN_RECORDS = 87_461
DEFAULT_TEST_RECORDS = 20_000
GENERATION_SENTENCE_TYPES = {"abstract", "normal paragraph"}
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
CONTENT_STOPWORDS = {
    "about",
    "after",
    "among",
    "associated",
    "before",
    "between",
    "compared",
    "during",
    "from",
    "have",
    "higher",
    "into",
    "more",
    "than",
    "that",
    "their",
    "these",
    "this",
    "those",
    "through",
    "using",
    "were",
    "with",
}


def iter_dataset_records(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield top-level ID/record pairs without loading the full JSON object."""
    with path.open("rb") as source:
        for record_id, record in ijson.kvitems(source, ""):
            if not isinstance(record_id, str) or not isinstance(record, dict):
                raise ValueError(f"{path}: invalid top-level record")
            yield record_id, record


@contextmanager
def atomic_text_writer(path: Path) -> Iterator[TextIO]:
    """Write beside the destination and replace it only after success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            yield output
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_line(output: TextIO, value: dict[str, Any]) -> None:
    output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    output.write("\n")


def _normalise_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def _normalise_sentence_type(sentence_type: str) -> str:
    return " ".join(sentence_type.replace("_", " ").split()).casefold()


def _text_digest(text: str) -> bytes:
    return hashlib.blake2b(
        _normalise_text(text).encode("utf-8"), digest_size=16
    ).digest()


def _candidate_pool_digest(
    sentences: list[str],
    sentence_types: list[str],
) -> bytes:
    payload = json.dumps(
        [sentences, sentence_types],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _content_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in WORD_RE.findall(text)
        if len(token) > 2 and token.casefold() not in CONTENT_STOPWORDS
    }


def _evidence_alignment(
    hypothesis: str,
    sentence: str,
    labels: list[str],
    aspect_mapping: dict[str, Any],
) -> int:
    hypothesis_tokens = _content_tokens(hypothesis)
    sentence_overlap = len(hypothesis_tokens & _content_tokens(sentence))
    aspect_overlap = 0
    for label in labels:
        aspect_text = aspect_mapping.get(label)
        if isinstance(aspect_text, str):
            aspect_overlap = max(
                aspect_overlap,
                len(hypothesis_tokens & _content_tokens(aspect_text)),
            )
    return sentence_overlap + aspect_overlap


def _validate_candidate_pool(
    source_path: Path,
    record_id: str,
    record: dict[str, Any],
) -> tuple[str, list[str], list[str]]:
    paper_id = record.get("paper_id")
    sentences = record.get("paper_as_candidate_pool")
    sentence_types = record.get("sentence_types_in_candidate_pool")
    if not isinstance(paper_id, str) or not paper_id:
        raise ValueError(f"{source_path}:{record_id}: invalid paper_id")
    if not isinstance(sentences, list) or not all(
        isinstance(sentence, str) for sentence in sentences
    ):
        raise ValueError(f"{source_path}:{record_id}: invalid candidate pool")
    if not isinstance(sentence_types, list) or not all(
        isinstance(sentence_type, str) for sentence_type in sentence_types
    ):
        raise ValueError(f"{source_path}:{record_id}: invalid sentence types")
    if len(sentences) != len(sentence_types):
        raise ValueError(
            f"{source_path}:{record_id}: candidate/type length mismatch "
            f"({len(sentences)} != {len(sentence_types)})"
        )
    return paper_id, sentences, sentence_types


def _corpus_row(
    paper_id: str,
    sentence_idx: int,
    sentence: str,
    sentence_type: str,
) -> dict[str, Any]:
    return {
        "doc_id": f"{paper_id}#{sentence_idx:06d}",
        "title": "",
        "text": sentence,
        "paper_id": paper_id,
        "sentence_idx": sentence_idx,
        "sentence_type": sentence_type,
        "split": "test",
    }


def _is_generation_candidate(
    sentence: str,
    sentence_type: str,
    min_generation_words: int,
) -> bool:
    return (
        _normalise_sentence_type(sentence_type) in GENERATION_SENTENCE_TYPES
        and bool(sentence.strip())
        and _word_count(sentence) >= min_generation_words
    )


def write_test_corpora(
    test_path: Path,
    full_output_path: Path,
    generation_output_path: Path,
    *,
    min_generation_words: int = 12,
) -> dict[str, int]:
    """Create the lossless test sentence corpus and clean generation subset."""
    if min_generation_words < 1:
        raise ValueError("min_generation_words must be positive")

    stats = {
        "records": 0,
        "unique_papers": 0,
        "duplicate_paper_records": 0,
        "sentences": 0,
        "generation_candidates": 0,
        "generation_duplicates_skipped": 0,
        "generation_ineligible_skipped": 0,
    }
    paper_pool_digests: dict[str, bytes] = {}
    seen_generation_texts: set[bytes] = set()

    with (
        atomic_text_writer(full_output_path) as full_output,
        atomic_text_writer(generation_output_path) as generation_output,
    ):
        for record_id, record in iter_dataset_records(test_path):
            paper_id, sentences, sentence_types = _validate_candidate_pool(
                test_path, record_id, record
            )
            stats["records"] += 1
            pool_digest = _candidate_pool_digest(sentences, sentence_types)
            prior_digest = paper_pool_digests.get(paper_id)
            if prior_digest is not None:
                if prior_digest != pool_digest:
                    raise ValueError(
                        f"{test_path}:{record_id}: repeated paper_id {paper_id!r} "
                        "has a different candidate pool"
                    )
                stats["duplicate_paper_records"] += 1
                continue
            paper_pool_digests[paper_id] = pool_digest
            stats["unique_papers"] += 1

            for sentence_idx, (sentence, sentence_type) in enumerate(
                zip(sentences, sentence_types)
            ):
                row = _corpus_row(
                    paper_id,
                    sentence_idx,
                    sentence,
                    sentence_type,
                )
                _write_json_line(full_output, row)
                stats["sentences"] += 1

                if not _is_generation_candidate(
                    sentence,
                    sentence_type,
                    min_generation_words,
                ):
                    stats["generation_ineligible_skipped"] += 1
                    continue

                digest = _text_digest(sentence)
                if digest in seen_generation_texts:
                    stats["generation_duplicates_skipped"] += 1
                    continue
                seen_generation_texts.add(digest)
                _write_json_line(generation_output, row)
                stats["generation_candidates"] += 1

    return stats


def _labels_for_sentence(
    sentence_index2aspects: dict[Any, Any],
    sentence_idx: int,
) -> list[str]:
    labels = sentence_index2aspects.get(str(sentence_idx))
    if labels is None:
        labels = sentence_index2aspects.get(sentence_idx)
    if not isinstance(labels, list):
        return []
    return [label for label in labels if isinstance(label, str) and label]


def _few_shot_candidate(
    source_path: Path,
    record_id: str,
    record: dict[str, Any],
) -> dict[str, Any] | None:
    hypothesis = record.get("hypothesis")
    review_id = record.get("systematic_review_id")
    sentence_index2aspects = record.get("sentence_index2aspects")
    aspect_mapping = record.get("aspect_id2aspect")
    if (
        not isinstance(hypothesis, str)
        or not hypothesis.strip()
        or not 8 <= _word_count(hypothesis) <= 35
        or review_id is None
        or not isinstance(sentence_index2aspects, dict)
        or not isinstance(aspect_mapping, dict)
    ):
        return None

    paper_id, sentences, sentence_types = _validate_candidate_pool(
        source_path, record_id, record
    )
    positive_options: list[dict[str, Any]] = []
    for sentence_idx, (sentence, sentence_type) in enumerate(
        zip(sentences, sentence_types)
    ):
        if _normalise_sentence_type(sentence_type) not in GENERATION_SENTENCE_TYPES:
            continue
        sentence_words = _word_count(sentence)
        if not 12 <= sentence_words <= 80:
            continue
        labels = _labels_for_sentence(sentence_index2aspects, sentence_idx)
        if not labels:
            continue
        positive_options.append(
            {
                "sentence": sentence.strip(),
                "sentence_type": sentence_type,
                "alignment": _evidence_alignment(
                    hypothesis,
                    sentence,
                    labels,
                    aspect_mapping,
                ),
                "coverage": len(set(labels)),
                "sentence_words": sentence_words,
            }
        )

    if not positive_options:
        return None

    positive = min(
        positive_options,
        key=lambda item: (
            -item["alignment"],
            -item["coverage"],
            abs(item["sentence_words"] - 35),
            item["sentence"],
        ),
    )
    return {
        "record_id": record_id,
        "paper_id": paper_id,
        "review_id": str(review_id),
        "hypothesis": hypothesis.strip(),
        "hypothesis_words": _word_count(hypothesis),
        **positive,
    }


def _select_distinct_candidates(
    candidates: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -item["alignment"],
            -item["coverage"],
            abs(item["sentence_words"] - 35),
            abs(item["hypothesis_words"] - 18),
            item["record_id"],
        ),
    )
    selected: list[dict[str, Any]] = []
    used_papers: set[str] = set()
    used_reviews: set[str] = set()
    used_queries: set[str] = set()
    used_positives: set[str] = set()

    def add_from(pool: list[dict[str, Any]], limit: int) -> None:
        for candidate in pool:
            if len(selected) >= limit:
                return
            query_key = _normalise_text(candidate["hypothesis"])
            positive_key = _normalise_text(candidate["sentence"])
            if (
                candidate["paper_id"] in used_papers
                or candidate["review_id"] in used_reviews
                or query_key in used_queries
                or positive_key in used_positives
            ):
                continue
            selected.append(candidate)
            used_papers.add(candidate["paper_id"])
            used_reviews.add(candidate["review_id"])
            used_queries.add(query_key)
            used_positives.add(positive_key)

    abstract_target = min(2, count)
    add_from(
        [
            item
            for item in ordered
            if _normalise_sentence_type(item["sentence_type"]) == "abstract"
        ],
        abstract_target,
    )
    add_from(ordered, count)
    return selected


def select_few_shots(
    train_path: Path,
    *,
    count: int = 5,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Select deterministic, traceable train-only hypothesis/evidence examples."""
    if count < 1:
        raise ValueError("count must be positive")

    stats = {
        "records": 0,
        "aspect_text_records": 0,
        "missing_aspect_text_records": 0,
        "nullable_results_aspect_records": 0,
        "nonnull_results_aspect_records": 0,
        "eligible_few_shot_records": 0,
    }
    candidates: list[dict[str, Any]] = []

    for record_id, record in iter_dataset_records(train_path):
        stats["records"] += 1
        aspect_mapping = record.get("aspect_id2aspect")
        if isinstance(aspect_mapping, dict) and aspect_mapping:
            stats["aspect_text_records"] += 1
        else:
            stats["missing_aspect_text_records"] += 1

        if (
            record.get("results_aspect_list_ids") is None
            and record.get("results_evidence_retrieval_at_optimal_evaluation") is None
        ):
            stats["nullable_results_aspect_records"] += 1
        else:
            stats["nonnull_results_aspect_records"] += 1

        candidate = _few_shot_candidate(train_path, record_id, record)
        if candidate is not None:
            candidates.append(candidate)
            stats["eligible_few_shot_records"] += 1

    selected = _select_distinct_candidates(candidates, count)
    if len(selected) != count:
        raise ValueError(
            f"{train_path}: requested {count} distinct few-shots, found {len(selected)}"
        )

    examples = [
        {
            "query": candidate["hypothesis"],
            "Positive": candidate["sentence"],
        }
        for candidate in selected
    ]
    return examples, stats


def write_json(path: Path, value: Any) -> None:
    with atomic_text_writer(path) as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a label-isolated EvidenceBench-100k test sentence corpus, "
            "generation subset, and train-only few-shot examples."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing the two official JSON files. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Derived output directory. Default: <dataset-dir>/derived",
    )
    parser.add_argument("--few-shot-count", type=int, default=5)
    parser.add_argument("--min-generation-words", type=int, default=12)
    parser.add_argument(
        "--expected-train-records",
        type=int,
        default=DEFAULT_TRAIN_RECORDS,
    )
    parser.add_argument(
        "--expected-test-records",
        type=int,
        default=DEFAULT_TEST_RECORDS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else dataset_dir / "derived"
    )
    train_path = dataset_dir / "evidencebench_100k_train_set.json"
    test_path = dataset_dir / "evidencebench_100k_test_set.json"
    for source_path in (train_path, test_path):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

    corpus_stats = write_test_corpora(
        test_path,
        output_dir / "evidencebench_100k_test_sentences.jsonl",
        output_dir / "evidencebench_100k_test_generation_candidates.jsonl",
        min_generation_words=args.min_generation_words,
    )
    examples, train_stats = select_few_shots(
        train_path,
        count=args.few_shot_count,
    )

    if corpus_stats["records"] != args.expected_test_records:
        raise ValueError(
            f"test record count: {corpus_stats['records']} != "
            f"{args.expected_test_records}"
        )
    if train_stats["records"] != args.expected_train_records:
        raise ValueError(
            f"train record count: {train_stats['records']} != "
            f"{args.expected_train_records}"
        )

    few_shot_path = output_dir / "evidencebench_100k_few_shot_examples.json"
    write_json(few_shot_path, examples)
    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "corpus": corpus_stats,
        "train": train_stats,
        "few_shot_count": len(examples),
        "min_generation_words": args.min_generation_words,
    }
    write_json(
        output_dir / "evidencebench_100k_conversion_stats.json",
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
