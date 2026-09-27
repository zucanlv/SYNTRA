import hashlib
import importlib.util
from pathlib import Path

import yaml


PIPELINE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PIPELINE_DIR
    / "config"
    / "RQ-2-generalization"
    / "medical"
    / "evidencebench"
    / "config_0805-evidencebench-10k.yaml"
)
CONFIG_TEMPLATE_PATH = PIPELINE_DIR / "config" / "config-example.yaml"
CORPUS_PATH = (
    "/data/share/project/shared_datasets/DSA/datasets/EvidenceBench-100k/"
    "derived/evidencebench_100k_test_sentences.jsonl"
)
EXPECTED_DEFINITION = (
    "Biomedical scientific hypothesis",
    "Sentence from a biomedical research paper",
    "Given a query (a biomedical scientific hypothesis) and a document "
    "(a sentence from a biomedical research paper), the document is relevant "
    "if it reports a finding, observation, analysis, or conclusion that directly "
    "supports, contradicts, qualifies, or otherwise provides evidence about the "
    "relationship or outcome asserted by the hypothesis. Merely mentioning the "
    "same biomedical entities or broad topic is insufficient.",
)
EXPECTED_FEW_SHOT_DIGESTS = [
    "2bdeef56f0206333a8a117eb0ea07398515eab154cdeebde2fd0a893b8c99d2e",
    "f8dec2fe650bb04a9747beca0cb00a7645aab8670ada4f34e6672ffe18ec953b",
    "50ea4826a42c736a24bedb889519d48ebd96de3edc68715893ab087e3d020f1d",
    "f35cfabb74fd836c12d8704c2978c15d2afd154fd29281a5e85db2f2020b8728",
    "61be330323b8a8a304653f57815a76ac48237c0c2509d4a877be6a9b5b366452",
]


def load_registry(filename, variable_name):
    path = PIPELINE_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, variable_name)


def test_evidencebench_high_level_definition():
    definitions = load_registry("HighLevel_Def.py", "HighLevel_Task_Definition")

    assert definitions["evidencebench"] == EXPECTED_DEFINITION


def test_evidencebench_uses_fixed_query_positive_examples():
    examples = load_registry("Few_Shot_Example.py", "Few_Shot_Example")
    task_examples = examples["evidencebench"]

    assert len(task_examples) == 5
    assert len({item["query"] for item in task_examples}) == 5
    assert all(set(item) == {"query", "Positive"} for item in task_examples)
    actual_digests = [
        hashlib.sha256(
            (item["query"] + "\0" + item["Positive"]).encode()
        ).hexdigest()
        for item in task_examples
    ]
    assert actual_digests == EXPECTED_FEW_SHOT_DIGESTS


def test_evidencebench_yaml_uses_sentence_corpus_and_safe_defaults():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["task"] == "evidencebench"
    assert config["corpus_path"] == CORPUS_PATH
    assert config["corpus_format"] == "jsonl"
    assert config["index_id_column"] == "doc_id"
    assert config["index_text_column"] == "text"
    assert config["index_title_column"] == ""
    assert config["diverse_texts"] is None
    assert config["skip_index"] is False
    assert config["production_sample"] == {
        "enabled": True,
        "mode": "random",
        "count": 10000,
        "seed": 42,
        "range": None,
        "indices": None,
    }
    assert config["llm_api_key"] is None
    assert "sk-" not in CONFIG_PATH.read_text(encoding="utf-8")


def test_evidencebench_yaml_keeps_the_complete_template_and_comments():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with open(CONFIG_TEMPLATE_PATH, "r", encoding="utf-8") as file:
        template = yaml.safe_load(file)

    assert set(template).issubset(config)
    assert set(config["llm_stages"]) == {
        "filter",
        "idattr_refine",
        "idattr_gen",
        "queryinstr",
        "diversequery",
        "annoinstr",
        "annotation",
        "queryfilter",
    }
    for stage_config in config["llm_stages"].values():
        assert set(stage_config) == {"model", "model_server", "api_key"}

    yaml_text = CONFIG_PATH.read_text(encoding="utf-8")
    required_comment_sections = [
        "# ===== Step 2.1 语料路径与格式 =====",
        "# ===== Step 2.2: 多样文本选取（DiverseSelection）专属配置 =====",
        "# ===== Step 4: 从外部 DiverseQuery 批处理切入（跳过 Step 2–3）=====",
        "# ===== LLM 客户端连接池（OpenAI-compatible / qwen-agent 基础设施）=====",
        "# ===== Step 5: 难负例挖掘配置（fully_annotation=false 时生效）=====",
        "# ===== Production mode 采样配置 =====",
        "# ===== AnnoInstr self-refine 数据源配置（仅 test_mode=true 且 fully_annotation=true 时生效）=====",
    ]
    assert all(section in yaml_text for section in required_comment_sections)
    assert sum(line.lstrip().startswith("#") for line in yaml_text.splitlines()) >= 100


if __name__ == "__main__":
    test_evidencebench_high_level_definition()
    test_evidencebench_uses_fixed_query_positive_examples()
    test_evidencebench_yaml_uses_sentence_corpus_and_safe_defaults()
    test_evidencebench_yaml_keeps_the_complete_template_and_comments()
    print("PASS: 4 tests")
