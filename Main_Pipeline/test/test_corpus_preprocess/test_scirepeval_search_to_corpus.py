import importlib.util
import json
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[2]
MODULE_PATH = PIPELINE_DIR / "corpus_preprocess" / "scirepeval_search_to_corpus.py"
CONFIG_PATH = (
    PIPELINE_DIR
    / "config"
    / "RQ-2-generalization"
    / "literature"
    / "config_0805-scirepeval-search.yaml"
)


def _load_module():
    assert MODULE_PATH.exists(), "SciRepEval Search corpus converter is missing"
    spec = importlib.util.spec_from_file_location(
        "scirepeval_search_to_corpus", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _candidate(doc_id: str, title: str, abstract: str, **metadata) -> dict:
    return {
        "doc_id": doc_id,
        "title": title,
        "abstract": abstract,
        "corpus_id": metadata.get("corpus_id", int(doc_id.removeprefix("d"))),
        "venue": metadata.get("venue", "Journal of Tests"),
        "year": metadata.get("year", 2024.0),
        "author_names": metadata.get("author_names", ["Test Author"]),
        "n_citations": metadata.get("n_citations", 3),
        "n_key_citations": metadata.get("n_key_citations", 1),
        "score": metadata.get("score", 1),
    }


def test_write_search_corpus_uses_official_search_document_fields(tmp_path):
    module = _load_module()
    output = tmp_path / "search-train-corpus.jsonl"
    long_abstract = (
        "This scientific paper presents a reproducible method and reports detailed "
        "experimental findings for a representative research problem."
    )
    rows = [
        {
            "query": "first query",
            "candidates": [
                _candidate("d1", "First paper", long_abstract, corpus_id=101),
                _candidate("d2", "Title-only paper", "Too short."),
            ],
        },
        {
            "query": "second query",
            "candidates": [
                _candidate("d1", "First paper", long_abstract, corpus_id=101),
                _candidate("d3", "Third paper", long_abstract, corpus_id=303),
            ],
        },
    ]

    stats = module.write_search_corpus(rows, output)

    assert _read_jsonl(output) == [
        {
            "doc_id": "d1",
            "title": "First paper",
            "abstract": long_abstract,
            "text": (
                "First paper </s> "
                f"{long_abstract} </s> Journal of Tests </s> 2024.0"
            ),
            "corpus_id": 101,
            "venue": "Journal of Tests",
            "year": 2024.0,
            "author_names": ["Test Author"],
            "n_citations": 3,
            "n_key_citations": 1,
            "split": "train",
        },
        {
            "doc_id": "d2",
            "title": "Title-only paper",
            "abstract": "Too short.",
            "text": (
                "Title-only paper </s> Too short. </s> "
                "Journal of Tests </s> 2024.0"
            ),
            "corpus_id": 2,
            "venue": "Journal of Tests",
            "year": 2024.0,
            "author_names": ["Test Author"],
            "n_citations": 3,
            "n_key_citations": 1,
            "split": "train",
        },
        {
            "doc_id": "d3",
            "title": "Third paper",
            "abstract": long_abstract,
            "text": (
                "Third paper </s> "
                f"{long_abstract} </s> Journal of Tests </s> 2024.0"
            ),
            "corpus_id": 303,
            "venue": "Journal of Tests",
            "year": 2024.0,
            "author_names": ["Test Author"],
            "n_citations": 3,
            "n_key_citations": 1,
            "split": "train",
        },
    ]
    assert stats == {
        "search_rows": 2,
        "candidates": 4,
        "documents_written": 3,
        "duplicate_doc_ids_skipped": 1,
    }


def test_official_document_rendering_skips_only_empty_fields():
    module = _load_module()
    candidate = _candidate(
        "d7",
        "Paper title",
        "Paper abstract",
        venue="",
        year=None,
    )

    assert module.render_search_document(candidate) == (
        "Paper title </s> Paper abstract"
    )


def test_search_assets_share_the_official_document_contract():
    namespace = {}
    exec((PIPELINE_DIR / "HighLevel_Def.py").read_text(encoding="utf-8"), namespace)
    exec((PIPELINE_DIR / "Few_Shot_Example.py").read_text(encoding="utf-8"), namespace)

    definition = namespace["HighLevel_Task_Definition"]["scirepeval-search"]
    examples = namespace["Few_Shot_Example"]["scirepeval-search"]

    assert definition[0] == "Scientific literature search query"
    assert definition[1] == (
        "Scientific paper title, abstract, venue, and publication year"
    )
    assert "title, abstract, venue, and publication year" in definition[2]
    assert len(examples) == 5
    assert all(example["query"].strip() for example in examples)
    assert all(example["Positive"].count(" </s> ") >= 2 for example in examples)
    assert all("Title: " not in example["Positive"] for example in examples)


def test_search_config_does_not_prepend_title_twice():
    import yaml

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["index_text_column"] == "text"
    assert config["index_title_column"] == ""
