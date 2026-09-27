#!/usr/bin/env python3
"""Convert SpartQA-MChoice and WinoGrande train sets to corpus/qpos artifacts.

The corpus contains every unique answer-option text. The qpos artifact uses the
pipeline's doc-centric schema and groups all unique original queries that share
the same correct answer document.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Iterable, Iterator


SPARTQA_INPUT = Path(
    "/data/share/project/shared_datasets/DSA/datasets/"
    "tasksource-spartqa-mchoice/spartqa_CO_train.jsonl"
)
SPARTQA_CORPUS_OUTPUT = SPARTQA_INPUT.parent / "corpus.jsonl"
SPARTQA_QPOS_OUTPUT = Path(
    "/data/share/project/shared_datasets/DSA/others/qpos/"
    "tasksource-spartqa-mchoice/spartqa_CO_train_qpos.json"
)

WINOGRANDE_INPUT = Path(
    "/data/share/project/shared_datasets/DSA/datasets/allenai-winogrande/"
    "winogrande_debiased/train-00000-of-00001.parquet"
)
WINOGRANDE_CORPUS_OUTPUT = WINOGRANDE_INPUT.parents[1] / "corpus.jsonl"
WINOGRANDE_QPOS_OUTPUT = Path(
    "/data/share/project/shared_datasets/DSA/others/qpos/"
    "allenai-winogrande/winogrande_debiased_train_qpos.json"
)
DEFAULT_SAMPLE_SEED = 42


def _required_text(value: object, *, field: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {field} must be a non-empty string")
    return value.strip()


def _candidate_texts(value: object, *, source: str) -> list[str]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{source}: candidate answers must contain at least two items")
    return [
        _required_text(item, field=f"candidate_answers[{index}]", source=source)
        for index, item in enumerate(value)
    ]


def parse_spartqa_row(row: dict, *, source: str) -> dict:
    """Map one SpartQA JSONL row to the converter's common record schema."""
    story = _required_text(row.get("story"), field="story", source=source)
    question = _required_text(row.get("question"), field="question", source=source)
    candidates = _candidate_texts(row.get("candidate_answers"), source=source)
    answer = row.get("answer")
    if type(answer) is not int or not 0 <= answer < len(candidates):
        raise ValueError(
            f"{source}: answer must be a zero-based candidate index, got {answer!r}"
        )
    return {
        "query": f"{story}\n\n{question}",
        "candidates": candidates,
        "correct_index": answer,
    }


def parse_winogrande_row(row: dict, *, source: str) -> dict:
    """Map one WinoGrande row to the converter's common record schema."""
    sentence = _required_text(row.get("sentence"), field="sentence", source=source)
    candidates = [
        _required_text(row.get("option1"), field="option1", source=source),
        _required_text(row.get("option2"), field="option2", source=source),
    ]
    answer = row.get("answer")
    if answer not in ("1", "2"):
        raise ValueError(f"{source}: answer must be '1' or '2', got {answer!r}")
    return {
        "query": sentence,
        "candidates": candidates,
        "correct_index": int(answer) - 1,
    }


def iter_spartqa_records(path: Path) -> Iterator[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"SpartQA train file not found: {path}")
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            source = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}: expected a JSON object")
            yield parse_spartqa_row(row, source=source)


def iter_winogrande_records(path: Path) -> Iterator[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"WinoGrande train file not found: {path}")
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    for row_index, row in enumerate(table.to_pylist()):
        yield parse_winogrande_row(row, source=f"{path}:{row_index}")


def build_artifacts(
    records: Iterable[dict],
    *,
    id_prefix: str,
    input_path: Path,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> tuple[list[dict], dict, dict]:
    """Build a deduplicated corpus with one sampled query per positive doc."""
    text_to_doc_id: dict[str, str] = {}
    corpus: list[dict] = []
    qpos_by_doc_id: dict[str, dict] = {}
    seen_pairs: set[tuple[str, str]] = set()
    query_to_positive: dict[str, str] = {}
    rows_read = 0
    candidate_occurrences = 0
    duplicate_pairs = 0

    for row_index, record in enumerate(records):
        source = f"record {row_index}"
        rows_read += 1
        query = _required_text(record.get("query"), field="query", source=source)
        candidates = _candidate_texts(record.get("candidates"), source=source)
        correct_index = record.get("correct_index")
        if type(correct_index) is not int or not 0 <= correct_index < len(candidates):
            raise ValueError(
                f"{source}: correct_index must select a candidate, got {correct_index!r}"
            )

        candidate_occurrences += len(candidates)
        for candidate in candidates:
            if candidate not in text_to_doc_id:
                doc_id = f"{id_prefix}{len(corpus):06d}"
                text_to_doc_id[candidate] = doc_id
                corpus.append({"_id": doc_id, "text": candidate})

        positive = candidates[correct_index]
        positive_doc_id = text_to_doc_id[positive]
        prior_positive = query_to_positive.get(query)
        if prior_positive is not None and prior_positive != positive_doc_id:
            raise ValueError(
                f"{source}: duplicate query maps to conflicting positive documents"
            )
        query_to_positive[query] = positive_doc_id

        pair = (query, positive_doc_id)
        if pair in seen_pairs:
            duplicate_pairs += 1
            continue
        seen_pairs.add(pair)

        result = qpos_by_doc_id.setdefault(
            positive_doc_id,
            {
                "doc_id": positive_doc_id,
                "doc": positive,
                "queries": [],
            },
        )
        result["queries"].append({"query": query})

    if not corpus:
        raise ValueError(f"no usable records found in {input_path}")

    rng = random.Random(sample_seed)
    results = list(qpos_by_doc_id.values())
    for result in results:
        result["queries"] = [rng.choice(result["queries"])]
    qpos_queries = sum(len(result["queries"]) for result in results)
    qpos = {
        "input_path": str(input_path.resolve()),
        "num_queries": 1,
        "results": results,
    }
    stats = {
        "rows_read": rows_read,
        "candidate_occurrences": candidate_occurrences,
        "corpus_documents": len(corpus),
        "qpos_results": len(results),
        "qpos_queries": qpos_queries,
        "duplicate_query_positive_pairs": duplicate_pairs,
    }
    return corpus, qpos, stats


def write_jsonl_atomic(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temp_path, path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def convert_one(
    records: Iterable[dict],
    *,
    input_path: Path,
    corpus_output: Path,
    qpos_output: Path,
    id_prefix: str,
) -> dict:
    corpus, qpos, stats = build_artifacts(
        records,
        id_prefix=id_prefix,
        input_path=input_path,
    )
    write_jsonl_atomic(corpus_output, corpus)
    write_json_atomic(qpos_output, qpos)
    return {
        **stats,
        "input": str(input_path),
        "corpus_output": str(corpus_output),
        "qpos_output": str(qpos_output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build answer-option corpora and train qpos for two MC datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "spartqa", "winogrande"),
        default="all",
        help="Dataset to convert. Default: all.",
    )
    parser.add_argument("--spartqa-input", type=Path, default=SPARTQA_INPUT)
    parser.add_argument(
        "--spartqa-corpus-output", type=Path, default=SPARTQA_CORPUS_OUTPUT
    )
    parser.add_argument("--spartqa-qpos-output", type=Path, default=SPARTQA_QPOS_OUTPUT)
    parser.add_argument("--winogrande-input", type=Path, default=WINOGRANDE_INPUT)
    parser.add_argument(
        "--winogrande-corpus-output", type=Path, default=WINOGRANDE_CORPUS_OUTPUT
    )
    parser.add_argument(
        "--winogrande-qpos-output", type=Path, default=WINOGRANDE_QPOS_OUTPUT
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {}
    if args.dataset in ("all", "spartqa"):
        summaries["spartqa"] = convert_one(
            iter_spartqa_records(args.spartqa_input),
            input_path=args.spartqa_input,
            corpus_output=args.spartqa_corpus_output,
            qpos_output=args.spartqa_qpos_output,
            id_prefix="spartqa-option-",
        )
    if args.dataset in ("all", "winogrande"):
        summaries["winogrande_debiased"] = convert_one(
            iter_winogrande_records(args.winogrande_input),
            input_path=args.winogrande_input,
            corpus_output=args.winogrande_corpus_output,
            qpos_output=args.winogrande_qpos_output,
            id_prefix="winogrande-debiased-option-",
        )
    json.dump(summaries, fp=os.sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
