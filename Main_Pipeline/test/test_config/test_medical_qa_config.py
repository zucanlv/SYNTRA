import importlib.util
import re
import sys
from pathlib import Path

import yaml


PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))

TEMPLATE_PATH = PIPELINE_DIR / "config/config-example.yaml"
CONFIG_PATH = (
    PIPELINE_DIR
    / "config/RQ-2-generalization/medical/medical_qa/config_0805-medical_qa-all.yaml"
)

from corpus_reader import CorpusConfig, CorpusReader


def mapping_key_paths(value, prefix=()):
    paths = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + (key,)
            paths.append(path)
            paths.extend(mapping_key_paths(child, path))
    return paths


def comment_suffixes(source):
    return [
        line[line.index("#") :] if "#" in line else None
        for line in source.splitlines()
    ]


def load_registry(filename, variable_name):
    path = PIPELINE_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, variable_name)


def test_medical_qa_config_is_runnable_and_matches_downloaded_corpus():
    template_source = TEMPLATE_PATH.read_text(encoding="utf-8")
    template = yaml.safe_load(template_source)
    source = CONFIG_PATH.read_text(encoding="utf-8")
    config = yaml.safe_load(source)

    assert mapping_key_paths(config) == mapping_key_paths(template)
    assert len(source.splitlines()) == len(template_source.splitlines())
    assert [
        index for index, line in enumerate(source.splitlines()) if not line
    ] == [
        index
        for index, line in enumerate(template_source.splitlines())
        if not line
    ]
    assert comment_suffixes(source) == comment_suffixes(template_source)

    assert config["task"] == "medical_qa"
    assert config["corpus_path"] == (
        "/data/share/project/shared_datasets/mteb/medical_qa/corpus.jsonl"
    )
    assert config["corpus_format"] == "jsonl"
    assert config["faiss_dir"] == (
        "/data/share/project/shared_datasets/DSA/faiss/medical_qa"
    )
    assert config["index_id_column"] == "_id"
    assert config["index_text_column"] == "text"
    assert config["index_title_column"] == "title"
    assert config["use_diverse_selection"] is False
    assert config["num_diverse_passages"] is None
    assert config["diverse_texts"] is None
    assert config["skip_index"] is True
    assert config["step4_diverse_query_input"] is None
    assert config["step5_query2passage_input"] is None
    assert config["test_mode"] is False
    assert config["production_sample"]["enabled"] is False
    assert config["anno_self_refine_source"] == "fewshot_positive"

    definitions = load_registry("HighLevel_Def.py", "HighLevel_Task_Definition")
    examples = load_registry("Few_Shot_Example.py", "Few_Shot_Example")
    assert "medical_qa" in definitions
    assert len(examples["medical_qa"]) == 5

    corpus_path = Path(config["corpus_path"])
    assert corpus_path.is_file()
    with corpus_path.open(encoding="utf-8") as corpus_file:
        assert sum(1 for line in corpus_file if line.strip()) == 2048

    assert config["llm_api_key"] is None
    assert re.search(
        r"sk-(?:or-)?[A-Za-z0-9_-]{16,}", source, flags=re.IGNORECASE
    ) is None

    reader = CorpusReader(
        CorpusConfig(
            corpus_path=config["corpus_path"],
            corpus_format=config["corpus_format"],
            id_column=config["index_id_column"],
            text_column=config["index_text_column"],
            title_column=config["index_title_column"],
        )
    )
    assert reader.column_map == {
        "doc_id": "_id",
        "text": "text",
        "title": "title",
    }


if __name__ == "__main__":
    test_medical_qa_config_is_runnable_and_matches_downloaded_corpus()
    print("PASS: medical_qa config")
