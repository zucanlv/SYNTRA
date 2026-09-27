from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Tuple


JsonDict = MutableMapping[str, object]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> Iterator[JsonDict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(obj).__name__}")
            yield obj


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_qrels(path: Path, qrels: Mapping[str, Mapping[str, int]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        for qid in sorted(qrels):
            for docid, score in sorted(qrels[qid].items()):
                writer.writerow([qid, docid, int(score)])


def read_beir_corpus(path: Path) -> Dict[str, str]:
    corpus_path = path / "corpus.jsonl"
    corpus: Dict[str, str] = {}
    for row in read_jsonl(corpus_path):
        doc_id = str(row.get("_id") or row.get("id") or "")
        if not doc_id:
            raise ValueError(f"Missing _id in {corpus_path}")
        title = str(row.get("title") or "")
        text = str(row.get("text") or "")
        corpus[doc_id] = (title + "\n" + text).strip()
    return corpus


def read_beir_queries(path: Path) -> Dict[str, str]:
    query_path = path / "queries.jsonl"
    queries: Dict[str, str] = {}
    for row in read_jsonl(query_path):
        query_id = str(row.get("_id") or row.get("id") or "")
        if not query_id:
            raise ValueError(f"Missing _id in {query_path}")
        queries[query_id] = str(row.get("text") or row.get("query") or "")
    return queries


def read_qrels(path: Path, split: str = "test") -> Dict[str, Dict[str, int]]:
    candidates = [
        path / "qrels" / f"{split}.tsv",
        path / "qrels" / f"{split}.txt",
        path / "qrels.tsv",
        path / "qrels.txt",
    ]
    qrels_path = next((p for p in candidates if p.exists()), None)
    if qrels_path is None:
        checked = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Could not find qrels file. Checked: {checked}")

    qrels: Dict[str, Dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return qrels
        parts = first.rstrip("\n").split("\t")
        has_header = parts[:3] == ["query-id", "corpus-id", "score"] or parts[:3] == ["qid", "docid", "score"]
        lines = f if has_header else [first, *f]
        for line_no, line in enumerate(lines, 2 if has_header else 1):
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) >= 3:
                qid, docid, score = cols[0], cols[1], cols[2]
            else:
                cols = line.split()
                if len(cols) == 4:
                    qid, _unused, docid, score = cols
                elif len(cols) >= 3:
                    qid, docid, score = cols[:3]
                else:
                    raise ValueError(f"Invalid qrels row at {qrels_path}:{line_no}: {line}")
            qrels.setdefault(str(qid), {})[str(docid)] = int(float(score))
    return qrels


def write_beir(
    output_dir: Path,
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    corpus_titles: Mapping[str, str] | None = None,
    split: str = "test",
) -> None:
    corpus_titles = corpus_titles or {}
    write_jsonl(
        output_dir / "corpus.jsonl",
        ({"_id": doc_id, "title": corpus_titles.get(doc_id, ""), "text": text} for doc_id, text in corpus.items()),
    )
    write_jsonl(output_dir / "queries.jsonl", ({"_id": qid, "text": text} for qid, text in queries.items()))
    write_qrels(output_dir / "qrels" / f"{split}.tsv", qrels)


def dedup_text_id(prefix: str, text: str, seen: Dict[str, str]) -> str:
    normalized = " ".join(str(text).split())
    if normalized in seen:
        return seen[normalized]
    doc_id = f"{prefix}{len(seen):08d}"
    seen[normalized] = doc_id
    return doc_id


def first_value(row: Mapping[str, object], aliases: Iterable[str]) -> object | None:
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def flatten_json_files(path: Path) -> Iterator[Tuple[Path, JsonDict]]:
    paths: list[Path]
    if path.is_file():
        paths = [path]
    else:
        paths = sorted([*path.rglob("*.jsonl"), *path.rglob("*.json")])
    for file_path in paths:
        if file_path.suffix == ".jsonl":
            yield from ((file_path, row) for row in read_jsonl(file_path))
        elif file_path.suffix == ".json":
            obj = read_json(file_path)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield file_path, item
            elif isinstance(obj, dict):
                for key in ("data", "examples", "questions", "instances", "rows"):
                    value = obj.get(key)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                yield file_path, item
                        break
                else:
                    yield file_path, obj
