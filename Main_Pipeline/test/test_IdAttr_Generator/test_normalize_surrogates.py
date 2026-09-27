import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


MAIN_PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN_PIPELINE_DIR))

from IdAttr_Generator import (  # noqa: E402
    IdAttrConfig,
    _normalize_surrogates,
    generate_idattr,
)


def test_normalize_surrogates_recursively_produces_utf8_safe_result():
    result = {
        "doc_id": "example",
        "doc": "ordinary text",
        "identifiers": [
            {
                "identifier": "math: \ud835\udc00",
                "label": "isolated: \ud835",
            }
        ],
        "attributes": [{"options": ["plain", "\udc00"]}],
    }

    normalized = _normalize_surrogates(result)
    serialized = json.dumps(normalized, ensure_ascii=False)

    assert normalized["identifiers"][0]["identifier"] == "math: \U0001d400"
    assert normalized["identifiers"][0]["label"] == "isolated: \ufffd"
    assert normalized["attributes"][0]["options"] == ["plain", "\ufffd"]
    assert serialized.encode("utf-8")


class FakeAssistant:
    def __init__(self, llm=None):
        self.llm = llm

    def run(self, messages):
        backslash = chr(92)
        yield [
            {
                "content": (
                    '{"identifiers": [{"identifier": "'
                    + backslash
                    + "ud835"
                    + backslash
                    + 'udc00"}], "attributes": []}'
                )
            }
        ]


def test_generate_idattr_normalizes_surrogates_before_writing():
    with TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.jsonl"
        input_path.write_text(
            json.dumps({"doc_id": "example", "doc": "ordinary text"}) + "\n",
            encoding="utf-8",
        )
        config = IdAttrConfig(
            input_path=str(input_path),
            task_name="bright-documents-theoremqa_theorems",
            output_dir=temp_dir,
            concurrency_enabled=False,
        )

        with patch("IdAttr_Generator.Assistant", FakeAssistant):
            output_path = generate_idattr(config)

        output = json.loads(Path(output_path).read_text(encoding="utf-8"))
        assert output["identifiers"][0]["identifier"] == "\U0001d400"


if __name__ == "__main__":
    test_normalize_surrogates_recursively_produces_utf8_safe_result()
    test_generate_idattr_normalizes_surrogates_before_writing()
