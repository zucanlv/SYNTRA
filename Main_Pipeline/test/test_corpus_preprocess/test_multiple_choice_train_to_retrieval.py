import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "corpus_preprocess"
    / "multiple_choice_train_to_retrieval.py"
)


def load_module():
    assert MODULE_PATH.is_file(), f"converter module does not exist: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location(
        "multiple_choice_train_to_retrieval", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spartqa_row_uses_story_question_and_zero_based_answer():
    module = load_module()

    record = module.parse_spartqa_row(
        {
            "story": "  A story.  ",
            "question": "  Which object?  ",
            "answer": 1,
            "candidate_answers": [" first ", " second "],
        },
        source="fixture.jsonl:1",
    )

    assert record == {
        "query": "A story.\n\nWhich object?",
        "candidates": ["first", "second"],
        "correct_index": 1,
    }


def test_winogrande_row_keeps_raw_sentence_and_converts_one_based_answer():
    module = load_module()

    record = module.parse_winogrande_row(
        {
            "sentence": "  The _ moved.  ",
            "option1": " Alice ",
            "option2": " Bob ",
            "answer": "2",
        },
        source="fixture.parquet:0",
    )

    assert record == {
        "query": "The _ moved.",
        "candidates": ["Alice", "Bob"],
        "correct_index": 1,
    }


def test_build_artifacts_samples_one_query_per_positive_doc():
    module = load_module()
    records = [
        {
            "query": "Question one",
            "candidates": [" Alpha ", "Beta"],
            "correct_index": 0,
        },
        {
            "query": "Question two",
            "candidates": ["Alpha", " Gamma "],
            "correct_index": 0,
        },
        {
            "query": "Question two",
            "candidates": ["Alpha", "Gamma"],
            "correct_index": 0,
        },
    ]

    corpus, qpos, stats = module.build_artifacts(
        records,
        id_prefix="demo-option-",
        input_path=Path("/tmp/train.data"),
    )

    assert corpus == [
        {"_id": "demo-option-000000", "text": "Alpha"},
        {"_id": "demo-option-000001", "text": "Beta"},
        {"_id": "demo-option-000002", "text": "Gamma"},
    ]
    assert qpos == {
        "input_path": "/tmp/train.data",
        "num_queries": 1,
        "results": [
            {
                "doc_id": "demo-option-000000",
                "doc": "Alpha",
                "queries": [{"query": "Question one"}],
            }
        ],
    }
    assert stats == {
        "rows_read": 3,
        "candidate_occurrences": 6,
        "corpus_documents": 3,
        "qpos_results": 1,
        "qpos_queries": 1,
        "duplicate_query_positive_pairs": 1,
    }


def test_invalid_answer_is_rejected_with_source_context():
    module = load_module()

    with unittest.TestCase().assertRaisesRegex(ValueError, r"fixture\.jsonl:7.*answer"):
        module.parse_spartqa_row(
            {
                "story": "Story",
                "question": "Question",
                "answer": 4,
                "candidate_answers": ["a", "b", "c", "d"],
            },
            source="fixture.jsonl:7",
        )


def test_atomic_writers_emit_jsonl_and_qpos_json(tmp_path):
    module = load_module()
    corpus_path = tmp_path / "corpus.jsonl"
    qpos_path = tmp_path / "qpos.json"
    corpus = [
        {"_id": "demo-option-000000", "text": "Alpha"},
        {"_id": "demo-option-000001", "text": "Beta"},
    ]
    qpos = {
        "input_path": "/tmp/train.data",
        "num_queries": 1,
        "results": [
            {
                "doc_id": "demo-option-000000",
                "doc": "Alpha",
                "queries": [{"query": "Question"}],
            }
        ],
    }

    module.write_jsonl_atomic(corpus_path, corpus)
    module.write_json_atomic(qpos_path, qpos)

    assert [
        json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()
    ] == corpus
    assert json.loads(qpos_path.read_text(encoding="utf-8")) == qpos


if __name__ == "__main__":
    test_spartqa_row_uses_story_question_and_zero_based_answer()
    test_winogrande_row_keeps_raw_sentence_and_converts_one_based_answer()
    test_build_artifacts_samples_one_query_per_positive_doc()
    test_invalid_answer_is_rejected_with_source_context()
    with tempfile.TemporaryDirectory() as temp_dir:
        test_atomic_writers_emit_jsonl_and_qpos_json(Path(temp_dir))
    print("PASS: 5 tests")
