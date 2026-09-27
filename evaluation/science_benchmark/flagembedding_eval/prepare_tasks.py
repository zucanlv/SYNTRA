from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Sequence

from .beir_io import dedup_text_id, first_value, flatten_json_files, read_json, read_jsonl, write_beir


TASK_ALIASES = {
    "kuake-qtr": "kuake",
    "financebench open-source subset": "evidence_json",
    "financebench": "evidence_json",
    "lofin benchmark": "beir_or_evidence_json",
    "lofin": "beir_or_evidence_json",
    "legalbench-rag": "evidence_json",
    "legalbench-rag cuad": "evidence_json",
    "legalcitebench citation retrieval": "evidence_json",
    "legalcitebench": "evidence_json",
    "scholarqabench": "beir_or_evidence_json",
    "scholarqabench search tasks": "beir_or_evidence_json",
    "evidencebench": "evidence_json",
    "kilt retrieval / provenance tasks": "evidence_json",
    "open rag benchmark": "evidence_json",
    "birco": "beir_or_evidence_json",
    "trialgpt retrieval benchmark": "evidence_json",
    "scirepeval search tasks": "beir_or_evidence_json",
    "lecard": "lecard",
    "lecardv2": "lecard",
    "generic_jsonl": "generic_jsonl",
}

QUERY_ALIASES = (
    "query",
    "question",
    "input",
    "claim",
    "hypothesis",
    "patient",
    "patient_summary",
    "patient_description",
    "summary",
    "case",
    "case_text",
    "citing_context",
    "citation_context",
    "problem",
    "text",
)
DOC_ALIASES = (
    "doc",
    "document",
    "passage",
    "context",
    "evidence",
    "gold_text",
    "answer",
    "snippet",
    "title",
    "positive",
    "positive_passage",
    "positive_doc",
    "positive_document",
    "gold",
    "golden",
    "gold_answer",
    "reference",
    "cited_case",
    "trial",
    "article",
    "paper",
)
DOC_ID_ALIASES = (
    "doc_id",
    "document_id",
    "corpus_id",
    "passage_id",
    "evidence_id",
    "pid",
    "pmid",
    "trial_id",
    "case_id",
    "paper_id",
    "article_id",
    "citation_id",
    "wikipedia_id",
    "id",
    "_id",
)
QUERY_ID_ALIASES = ("query_id", "qid", "id", "_id", "question_id", "sample_id", "problem_id")
LABEL_ALIASES = ("label", "score", "relevance", "relevant", "is_relevant", "rating", "grade")
POSITIVE_ID_KEYS = (
    "positive_doc_ids",
    "positive_document_ids",
    "positive_passage_ids",
    "relevant_doc_ids",
    "relevant_docs",
    "gold_doc_ids",
    "gold_docs",
    "answer_doc_ids",
    "evidence_doc_ids",
    "cited_doc_ids",
    "citation_ids",
)
CANDIDATE_LIST_KEYS = (
    "positive_docs",
    "positive_documents",
    "positive_passages",
    "positives",
    "gold_docs",
    "gold_documents",
    "evidences",
    "evidence",
    "snippets",
    "contexts",
    "documents",
    "docs",
    "passages",
    "candidates",
    "candidate_docs",
    "candidate_documents",
    "candidate_passages",
    "retrieved_docs",
    "trials",
    "citations",
    "cited_cases",
    "papers",
)
NEGATIVE_LIST_KEYS = ("negative_docs", "negative_documents", "negative_passages", "negatives", "hard_negatives")


def norm_task(task: str) -> str:
    return " ".join(task.strip().lower().split())


def maybe_copy_beir(input_dir: Path, output_dir: Path) -> bool:
    if not input_dir.is_dir():
        return False
    if (input_dir / "corpus.jsonl").exists() and (input_dir / "queries.jsonl").exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_dir / "corpus.jsonl", output_dir / "corpus.jsonl")
        shutil.copy2(input_dir / "queries.jsonl", output_dir / "queries.jsonl")
        if (input_dir / "qrels").exists():
            if (output_dir / "qrels").exists():
                shutil.rmtree(output_dir / "qrels")
            shutil.copytree(input_dir / "qrels", output_dir / "qrels")
        else:
            for name in ("qrels.tsv", "qrels.txt"):
                if (input_dir / name).exists():
                    shutil.copy2(input_dir / name, output_dir / name)
        return True
    return False


def textify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(textify(v) for v in value if textify(v))
    if isinstance(value, dict):
        parts = []
        for key in (
            "title",
            "text",
            "content",
            "contents",
            "abstract",
            "body",
            "answer",
            "evidence",
            "snippet",
            "summary",
            "description",
            "judgment",
            "judgement",
            "fact",
            "facts",
        ):
            if key in value:
                part = textify(value[key])
                if part:
                    parts.append(part)
        if parts:
            return "\n".join(parts)
        return "\n".join(f"{k}: {textify(v)}" for k, v in value.items() if textify(v))
    return str(value)


def parse_relevance(value: object, default: int = 1) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return default if value else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "relevant", "positive", "entailment", "correct"}:
            return default
        if lowered in {"false", "no", "irrelevant", "negative", "not_relevant", "incorrect"}:
            return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def read_table(path: Path) -> Iterator[dict[str, object]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.readline()
        if not sample:
            return
        f.seek(0)
        cells = sample.rstrip("\n").split(delimiter)
        known_headers = {
            "query-id",
            "corpus-id",
            "qid",
            "docid",
            "doc_id",
            "query_id",
            "score",
            "text",
            "query",
            "document",
            "title",
        }
        has_header = any(cell.strip() in known_headers for cell in cells)
        if has_header:
            yield from csv.DictReader(f, delimiter=delimiter)
            return
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) == 2:
                yield {"id": row[0], "text": row[1]}
            elif len(row) == 3:
                yield {"qid": row[0], "doc_id": row[1], "score": row[2]}
            elif len(row) >= 4:
                yield {"qid": row[0], "doc_id": row[2], "score": row[3]}


def read_records(path: Path) -> Iterator[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from read_jsonl(path)
    elif suffix == ".json":
        obj = read_json(path)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
        elif isinstance(obj, dict):
            for key in ("data", "examples", "questions", "instances", "rows", "queries", "documents", "corpus"):
                value = obj.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                    return
            yield obj
    elif suffix in {".csv", ".tsv", ".txt"}:
        yield from read_table(path)
    else:
        raise ValueError(f"Unsupported file extension for {path}")


def first_existing_file(input_dir: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = input_dir / name
        if candidate.exists():
            return candidate
    return None


def add_doc(corpus: Dict[str, str], seen_docs: Dict[str, str], explicit_doc_id: object, doc_text: str) -> str:
    doc_id = str(explicit_doc_id) if explicit_doc_id not in (None, "") else dedup_text_id("d", doc_text, seen_docs)
    corpus.setdefault(doc_id, doc_text)
    return doc_id


def doc_id_from_item(item: object) -> object | None:
    if isinstance(item, dict):
        return first_value(item, DOC_ID_ALIASES)
    return item if isinstance(item, (str, int)) else None


def iter_candidate_docs(row: Mapping[str, object]) -> Iterable[tuple[str | None, str, int]]:
    """Yield (doc_id, doc_text, relevance) triples from common evidence schemas."""
    direct_doc = first_value(row, DOC_ALIASES)
    direct_doc_id = first_value(row, DOC_ID_ALIASES)
    label = first_value(row, LABEL_ALIASES)
    relevance = parse_relevance(label)
    if direct_doc:
        yield (str(direct_doc_id) if direct_doc_id else None, textify(direct_doc), relevance)

    for key in CANDIDATE_LIST_KEYS:
        value = row.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                doc_id = first_value(item, DOC_ID_ALIASES)
                doc_text = first_value(item, DOC_ALIASES) or item
                score = first_value(item, LABEL_ALIASES)
                yield (str(doc_id) if doc_id else None, textify(doc_text), parse_relevance(score))
            else:
                yield None, textify(item), 1

    for key in NEGATIVE_LIST_KEYS:
        value = row.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                doc_id = first_value(item, DOC_ID_ALIASES)
                doc_text = first_value(item, DOC_ALIASES) or item
                yield (str(doc_id) if doc_id else None, textify(doc_text), 0)
            else:
                yield None, textify(item), 0


def iter_positive_doc_ids(row: Mapping[str, object]) -> Iterable[str]:
    for key in POSITIVE_ID_KEYS:
        for item in as_list(row.get(key)):
            doc_id = doc_id_from_item(item)
            if doc_id not in (None, ""):
                yield str(doc_id)
    labels = first_value(row, ("qrels", "labels", "golden_labels", "relevance", "relevant_documents"))
    if isinstance(labels, dict):
        for doc_id, score in labels.items():
            if parse_relevance(score) > 0:
                yield str(doc_id)
    elif isinstance(labels, list):
        for item in labels:
            if isinstance(item, dict):
                doc_id = first_value(item, DOC_ID_ALIASES)
                score = first_value(item, LABEL_ALIASES)
                if doc_id not in (None, "") and parse_relevance(score) > 0:
                    yield str(doc_id)
            elif item not in (None, ""):
                yield str(item)


def iter_kilt_provenance(row: Mapping[str, object]) -> Iterable[tuple[str, str]]:
    for output in as_list(row.get("output")):
        if not isinstance(output, dict):
            continue
        for provenance in as_list(output.get("provenance")):
            if not isinstance(provenance, dict):
                continue
            wikipedia_id = provenance.get("wikipedia_id")
            title = provenance.get("title") or provenance.get("wikipedia_title")
            start = provenance.get("start_paragraph_id")
            end = provenance.get("end_paragraph_id")
            doc_id = ":".join(str(v) for v in (wikipedia_id or title, start, end) if v not in (None, ""))
            doc_text = textify(provenance)
            if doc_id and doc_text:
                yield doc_id, doc_text


def convert_evidence_json(input_path: Path, output_dir: Path) -> None:
    if maybe_copy_beir(input_path, output_dir):
        return

    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}
    seen_docs: Dict[str, str] = {}

    for row_idx, (_path, row) in enumerate(flatten_json_files(input_path)):
        query = first_value(row, QUERY_ALIASES)
        if not query:
            continue
        qid = str(first_value(row, QUERY_ID_ALIASES) or f"q{row_idx:08d}")
        queries[qid] = textify(query)

        found = False
        for explicit_doc_id, doc_text, relevance in iter_candidate_docs(row):
            if not doc_text:
                continue
            doc_id = add_doc(corpus, seen_docs, explicit_doc_id, doc_text)
            if relevance > 0:
                qrels.setdefault(qid, {})[doc_id] = relevance
            found = True
        for doc_id in iter_positive_doc_ids(row):
            if doc_id in corpus:
                qrels.setdefault(qid, {})[doc_id] = max(qrels.setdefault(qid, {}).get(doc_id, 0), 1)
                found = True
        for doc_id, doc_text in iter_kilt_provenance(row):
            corpus.setdefault(doc_id, doc_text)
            qrels.setdefault(qid, {})[doc_id] = 1
            found = True
        if not found:
            answer = first_value(row, ("gold", "gold_answer", "gold_text", "answer", "answers"))
            if answer:
                doc_text = textify(answer)
                doc_id = dedup_text_id("d", doc_text, seen_docs)
                corpus.setdefault(doc_id, doc_text)
                qrels.setdefault(qid, {})[doc_id] = 1

    if not queries or not corpus or not qrels:
        raise ValueError(
            f"Could not infer query/corpus/qrels from {input_path}. "
            "Use generic_jsonl with explicit --query-field/--doc-field/--label-field, "
            "or provide --corpus-file/--queries-file/--qrels-file."
        )
    write_beir(output_dir, corpus, queries, qrels)


def convert_generic_jsonl(
    input_file: Path,
    output_dir: Path,
    query_field: str,
    doc_field: str,
    label_field: str | None,
    query_id_field: str | None,
    doc_id_field: str | None,
) -> None:
    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}
    seen_docs: Dict[str, str] = {}

    for idx, row in enumerate(read_jsonl(input_file)):
        query = textify(row.get(query_field))
        doc = textify(row.get(doc_field))
        if not query or not doc:
            continue
        qid = str(row.get(query_id_field) if query_id_field else f"q{idx:08d}")
        doc_id = str(row.get(doc_id_field)) if doc_id_field and row.get(doc_id_field) else dedup_text_id("d", doc, seen_docs)
        score = parse_relevance(row.get(label_field) if label_field else 1)
        queries[qid] = query
        corpus.setdefault(doc_id, doc)
        if score > 0:
            qrels.setdefault(qid, {})[doc_id] = score
    write_beir(output_dir, corpus, queries, qrels)


def convert_explicit_ir_files(
    output_dir: Path,
    corpus_file: Path,
    queries_file: Path,
    qrels_file: Path,
    query_field: str | None = None,
    doc_field: str | None = None,
    label_field: str | None = None,
    query_id_field: str | None = None,
    doc_id_field: str | None = None,
) -> None:
    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    for idx, row in enumerate(read_records(corpus_file)):
        doc_id = first_value(row, (doc_id_field,)) if doc_id_field else first_value(row, DOC_ID_ALIASES)
        doc_text = textify(row.get(doc_field)) if doc_field else textify(first_value(row, DOC_ALIASES) or row)
        if doc_text:
            corpus[str(doc_id or f"d{idx:08d}")] = doc_text

    for idx, row in enumerate(read_records(queries_file)):
        qid = first_value(row, (query_id_field,)) if query_id_field else first_value(row, QUERY_ID_ALIASES)
        query = textify(row.get(query_field)) if query_field else textify(first_value(row, QUERY_ALIASES) or row)
        if query:
            queries[str(qid or f"q{idx:08d}")] = query

    for row in read_records(qrels_file):
        qid = first_value(row, tuple(k for k in (query_id_field, "query-id", "query_id", "qid", "question_id", "id") if k))
        doc_id = first_value(row, tuple(k for k in (doc_id_field, "corpus-id", "doc_id", "document_id", "passage_id", "pid", "case_id", "trial_id") if k))
        score = first_value(row, tuple(k for k in (label_field, "score", "relevance", "label", "rating", "grade") if k))
        if qid in (None, "") or doc_id in (None, ""):
            continue
        rel = parse_relevance(score)
        if rel > 0:
            qrels.setdefault(str(qid), {})[str(doc_id)] = rel

    if not corpus or not queries or not qrels:
        raise ValueError("Explicit IR conversion requires non-empty corpus, queries, and qrels.")
    write_beir(output_dir, corpus, queries, qrels)


def convert_auto_ir_dir(input_dir: Path, output_dir: Path) -> bool:
    if not input_dir.is_dir():
        return False
    corpus_file = first_existing_file(
        input_dir,
        (
            "corpus.jsonl",
            "corpus.json",
            "documents.jsonl",
            "documents.json",
            "docs.jsonl",
            "docs.json",
            "collection.tsv",
            "corpus.tsv",
        ),
    )
    queries_file = first_existing_file(
        input_dir,
        ("queries.jsonl", "queries.json", "questions.jsonl", "questions.json", "topics.tsv", "queries.tsv"),
    )
    qrels_file = first_existing_file(
        input_dir,
        ("qrels/test.tsv", "qrels.tsv", "qrels.txt", "labels.tsv", "relevance.tsv", "test_qrels.tsv"),
    )
    if not corpus_file or not queries_file or not qrels_file:
        return False
    convert_explicit_ir_files(output_dir, corpus_file, queries_file, qrels_file)
    return True


def convert_kuake(input_dir: Path, output_dir: Path) -> None:
    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}
    seen_docs: Dict[str, str] = {}
    files = sorted([*input_dir.glob("*.json"), *input_dir.glob("*.jsonl")])
    if not files:
        raise FileNotFoundError(f"No KUAKE json/jsonl files found in {input_dir}")
    for file_path in files:
        rows = read_json(file_path) if file_path.suffix == ".json" else list(read_jsonl(file_path))
        if isinstance(rows, dict):
            rows = rows.get("data") or rows.get("examples") or []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            query = textify(first_value(row, ("query", "question", "sentence1")))
            title = textify(first_value(row, ("title", "sentence2", "text")))
            if not query or not title:
                continue
            qid = str(first_value(row, ("id", "qid", "query_id")) or f"{file_path.stem}-q{idx:08d}")
            doc_id = dedup_text_id("t", title, seen_docs)
            score = parse_relevance(first_value(row, ("label", "score", "relevance")), default=0)
            queries[qid] = query
            corpus.setdefault(doc_id, title)
            if score > 0:
                qrels.setdefault(qid, {})[doc_id] = score
    write_beir(output_dir, corpus, queries, qrels)


def stringify_case(value: object) -> str:
    if isinstance(value, dict):
        parts = []
        for key in ("title", "fact", "facts", "reason", "judgement", "judgment", "text", "content"):
            if key in value:
                part = textify(value[key])
                if part:
                    parts.append(part)
        return "\n".join(parts) if parts else textify(value)
    return textify(value)


def convert_lecard(input_dir: Path, output_dir: Path) -> None:
    if maybe_copy_beir(input_dir, output_dir):
        return
    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    for _path, row in flatten_json_files(input_dir):
        qid = str(first_value(row, ("qid", "query_id", "id")) or "")
        query = first_value(row, ("query", "query_case", "case", "fact", "facts", "text", "content"))
        candidates = first_value(row, ("candidates", "candidate_cases", "docs", "documents"))
        labels = first_value(row, ("labels", "golden_labels", "qrels", "relevance"))
        if qid and query:
            queries[qid] = stringify_case(query)
        if isinstance(candidates, list):
            for idx, candidate in enumerate(candidates):
                doc_id = str(first_value(candidate, ("doc_id", "case_id", "id", "cid")) if isinstance(candidate, dict) else f"{qid}-d{idx:06d}")
                corpus.setdefault(doc_id, stringify_case(candidate))
        if qid and isinstance(labels, dict):
            for doc_id, score in labels.items():
                rel = parse_relevance(score)
                if rel > 0:
                    qrels.setdefault(qid, {})[str(doc_id)] = rel
        elif qid and isinstance(labels, list):
            for item in labels:
                if isinstance(item, dict):
                    doc_id = first_value(item, ("doc_id", "case_id", "id", "cid"))
                    score = first_value(item, ("score", "label", "relevance"))
                    if doc_id:
                        rel = parse_relevance(score)
                        if rel > 0:
                            qrels.setdefault(qid, {})[str(doc_id)] = rel
                else:
                    qrels.setdefault(qid, {})[str(item)] = 1

    if not queries or not corpus or not qrels:
        raise ValueError(f"Could not infer LeCaRD layout from {input_dir}. Convert to BEIR manually or use generic_jsonl.")
    write_beir(output_dir, corpus, queries, qrels)


def convert_beir_or_evidence(input_dir: Path, output_dir: Path) -> None:
    if not maybe_copy_beir(input_dir, output_dir) and not convert_auto_ir_dir(input_dir, output_dir):
        convert_evidence_json(input_dir, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert retrieval datasets to BEIR layout for FlagEmbedding evaluation.")
    parser.add_argument("--task", required=True, help="Task name from benchmark.csv or a preset name.")
    parser.add_argument("--input-dir", type=Path, help="Raw task directory.")
    parser.add_argument("--input-file", type=Path, help="Raw task JSONL file for generic_jsonl.")
    parser.add_argument("--hf-dataset", help="Optional HuggingFace dataset name to cache and convert.")
    parser.add_argument("--hf-config", help="Optional HuggingFace dataset config.")
    parser.add_argument("--hf-split", default="test", help="HuggingFace split to load when --hf-dataset is used.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--query-field")
    parser.add_argument("--doc-field")
    parser.add_argument("--label-field")
    parser.add_argument("--query-id-field")
    parser.add_argument("--doc-id-field")
    parser.add_argument("--corpus-file", type=Path, help="Explicit corpus file for non-standard IR layouts.")
    parser.add_argument("--queries-file", type=Path, help="Explicit queries file for non-standard IR layouts.")
    parser.add_argument("--qrels-file", type=Path, help="Explicit qrels file for non-standard IR layouts.")
    args = parser.parse_args()

    task_key = TASK_ALIASES.get(norm_task(args.task), norm_task(args.task))
    input_path = args.input_file or args.input_dir
    if args.hf_dataset:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit("datasets is required for --hf-dataset. Install with: pip install datasets") from exc
        dataset = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
        cache_file = args.output_dir / "_source_hf_cache.jsonl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        input_path = cache_file
    if input_path is None and not (args.corpus_file and args.queries_file and args.qrels_file):
        raise SystemExit("--input-dir, --input-file, --hf-dataset, or explicit --corpus-file/--queries-file/--qrels-file is required")

    if args.corpus_file or args.queries_file or args.qrels_file:
        if not args.corpus_file or not args.queries_file or not args.qrels_file:
            raise SystemExit("--corpus-file, --queries-file, and --qrels-file must be provided together")
        convert_explicit_ir_files(
            args.output_dir,
            args.corpus_file,
            args.queries_file,
            args.qrels_file,
            query_field=args.query_field,
            doc_field=args.doc_field,
            label_field=args.label_field,
            query_id_field=args.query_id_field,
            doc_id_field=args.doc_id_field,
        )
    elif task_key == "kuake":
        convert_kuake(input_path, args.output_dir)
    elif task_key == "lecard":
        convert_lecard(input_path, args.output_dir)
    elif task_key == "evidence_json":
        convert_evidence_json(input_path, args.output_dir)
    elif task_key == "beir_or_evidence_json":
        convert_beir_or_evidence(input_path, args.output_dir)
    elif task_key == "generic_jsonl":
        if not args.query_field or not args.doc_field:
            raise SystemExit("generic_jsonl requires --query-field and --doc-field")
        convert_generic_jsonl(
            input_path,
            args.output_dir,
            args.query_field,
            args.doc_field,
            args.label_field,
            args.query_id_field,
            args.doc_id_field,
        )
    else:
        raise SystemExit(f"Unknown task preset: {args.task}. Available: {', '.join(sorted(TASK_ALIASES))}")

    print(f"Wrote BEIR-style task to {args.output_dir}")


if __name__ == "__main__":
    main()
