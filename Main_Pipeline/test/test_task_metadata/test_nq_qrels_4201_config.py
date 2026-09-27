import json
from pathlib import Path

import yaml


PIPELINE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PIPELINE_DIR
    / "config/RQ-2-generalization/nq/config_0811-nq-test-qrels-4201.yaml"
)
QRELS_PATH = Path(
    "/data/share/project/shared_datasets/DSA/datasets/mteb-nq/qrels/test.jsonl"
)
CORPUS_PATH = Path(
    "/data/share/project/shared_datasets/DSA/datasets/mteb-nq/"
    "nq-test-qrels-positive-corpus-4201.jsonl"
)


def _jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def test_nq_qrels_config_uses_the_complete_positive_corpus():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["task"] == "nq"
    assert Path(config["corpus_path"]) == CORPUS_PATH
    assert config["faiss_dir"].endswith("/nq")
    assert Path(config["diverse_texts"]).name == "nq-test-qrels-positive-diverse-texts-4201.jsonl"
    assert config["skip_index"] is True
    assert config["use_diverse_selection"] is False
    assert config["production_sample"]["enabled"] is False


def test_nq_qrels_corpus_exactly_matches_unique_test_positive_ids():
    expected_ids = {str(row["corpus-id"]) for row in _jsonl_rows(QRELS_PATH)}
    corpus_rows = list(_jsonl_rows(CORPUS_PATH))
    actual_ids = [str(row["_id"]) for row in corpus_rows]

    assert len(expected_ids) == 4201
    assert len(actual_ids) == 4201
    assert len(set(actual_ids)) == 4201
    assert set(actual_ids) == expected_ids


if __name__ == "__main__":
    test_nq_qrels_config_uses_the_complete_positive_corpus()
    test_nq_qrels_corpus_exactly_matches_unique_test_positive_ids()
    print("PASS: NQ qrels-4201 config and corpus")
