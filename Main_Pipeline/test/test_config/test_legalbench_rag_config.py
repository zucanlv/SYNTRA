import importlib.util
import re
import sys
from pathlib import Path

import yaml


PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))

CONFIG_PATH = (
    PIPELINE_DIR
    / "config/RQ-2-generalization/law/legalbench-rag/config_0805-legalbench-rag-test.yaml"
)
TEMPLATE_PATH = PIPELINE_DIR / "config/config-example.yaml"

from corpus_reader import CorpusConfig, CorpusReader


def load_registry(filename, variable_name):
    path = PIPELINE_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, variable_name)


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


def test_legalbench_rag_test_config_is_complete_and_runnable():
    config_source = CONFIG_PATH.read_text(encoding="utf-8")
    template_source = TEMPLATE_PATH.read_text(encoding="utf-8")
    config = yaml.safe_load(config_source)
    template = yaml.safe_load(template_source)

    definitions = load_registry("HighLevel_Def.py", "HighLevel_Task_Definition")
    examples = load_registry("Few_Shot_Example.py", "Few_Shot_Example")

    assert mapping_key_paths(config) == mapping_key_paths(template)
    assert len(config_source.splitlines()) == len(template_source.splitlines())
    assert [
        index for index, line in enumerate(config_source.splitlines()) if not line
    ] == [
        index for index, line in enumerate(template_source.splitlines()) if not line
    ]
    assert comment_suffixes(config_source) == comment_suffixes(template_source)
    assert config["task"] == "legalbench-rag"
    assert "legalbench-rag" in definitions
    assert len(examples["legalbench-rag"]) == 8

    assert Path(config["corpus_path"]).is_file()
    assert config["corpus_format"] == "jsonl"
    assert config["index_id_column"] == "id"
    assert config["index_text_column"] == "text"
    assert config["index_title_column"] == ""
    assert config["use_diverse_selection"] is False
    assert config["num_diverse_passages"] is None
    assert config["diverse_texts"] is None
    assert config["skip_index"] is True

    assert config["test_mode"] is True
    assert config["test_sample_size"] == 5
    assert config["query_filter_golden_query_count"] == 8
    assert config["anno_self_refine_source"] == "fewshot_positive"
    assert config["anno_fewshot_calibration_count"] is None
    assert config["production_sample"]["enabled"] is False

    assert config["llm_api_key"] is None
    assert re.search(
        r"sk-(?:or-)?[A-Za-z0-9_-]{16,}", config_source, flags=re.IGNORECASE
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
    assert reader.column_map == {"doc_id": "id", "text": "text", "title": None}


if __name__ == "__main__":
    test_legalbench_rag_test_config_is_complete_and_runnable()
    print("PASS: legalbench-rag test config")
