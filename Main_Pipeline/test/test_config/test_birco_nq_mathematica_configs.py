import importlib.util
import re
import sys
from pathlib import Path

import yaml


PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))

TEMPLATE_PATH = PIPELINE_DIR / "config/config-example.yaml"

from corpus_reader import CorpusConfig, CorpusReader


CONFIGS = {
    "birco-relic": {
        "path": PIPELINE_DIR
        / "config/RQ-2-generalization/literature/config_0801-birco-relic-all.yaml",
        "corpus_format": "parquet",
        "title_column": "",
        "use_diverse_selection": False,
        "num_diverse_passages": None,
    },
    "cqadupstack-mathematica": {
        "path": PIPELINE_DIR
        / "config/RQ-2-generalization/math/config_0801-cqadupstack-mathematica-10k.yaml",
        "corpus_format": "jsonl",
        "title_column": "title",
        "use_diverse_selection": True,
        "num_diverse_passages": 10000,
    },
    "nq": {
        "path": PIPELINE_DIR
        / "config/RQ-2-generalization/nq/config_0801-nq-10k.yaml",
        "corpus_format": "jsonl",
        "title_column": "title",
        "use_diverse_selection": True,
        "num_diverse_passages": 10000,
    },
}


def mapping_key_paths(value, prefix=()):
    """Return every mapping key path in insertion order."""
    paths = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + (key,)
            paths.append(path)
            paths.extend(mapping_key_paths(child, path))
    return paths


def comment_suffixes(source):
    """Keep comment text and its exact line position for comparison."""
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


def test_configs_are_runnable_and_match_downloaded_corpora():
    definitions = load_registry("HighLevel_Def.py", "HighLevel_Task_Definition")
    examples = load_registry("Few_Shot_Example.py", "Few_Shot_Example")
    template_source = TEMPLATE_PATH.read_text(encoding="utf-8")
    template = yaml.safe_load(template_source)

    for task, expected in CONFIGS.items():
        config_path = expected["path"]
        source = config_path.read_text(encoding="utf-8")
        config = yaml.safe_load(source)

        # These configs must be value-specialized copies of config-example:
        # preserve every field, nested field, order, blank line, and comment.
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

        assert config["task"] == task
        assert task in definitions
        assert len(examples[task]) == 5
        assert config["corpus_format"] == expected["corpus_format"]
        assert config["index_title_column"] == expected["title_column"]
        assert config["use_diverse_selection"] is expected["use_diverse_selection"]
        assert config["num_diverse_passages"] == expected["num_diverse_passages"]
        assert Path(config["corpus_path"]).is_file()
        assert config["skip_index"] is True
        assert config["step4_diverse_query_input"] is None
        assert config["step5_query2passage_input"] is None
        assert config["test_mode"] is False
        assert config["production_sample"]["enabled"] is False
        assert config["anno_self_refine_source"] == "fewshot_positive"

        # External credentials must come from OPENAI_API_KEY, never from YAML.
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
        columns = reader.column_map
        assert columns["doc_id"] == "_id"
        assert columns["text"] == "text"
        assert columns["title"] == (expected["title_column"] or None)


if __name__ == "__main__":
    test_configs_are_runnable_and_match_downloaded_corpora()
    print("PASS: 3 configs")
