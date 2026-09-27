from pathlib import Path
import sys
from unittest.mock import patch


MAIN_PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN_PIPELINE_DIR))

from Batch_Candidate_Annotator import BatchCandidateAnnotator  # noqa: E402


class FakeAssistant:
    last_messages = None

    def __init__(self, llm=None, system_message=None):
        self.llm = llm
        self.system_message = system_message

    def run(self, messages):
        FakeAssistant.last_messages = messages
        yield [
            {
                "content": (
                    '{"score": 1, "classification": "hard_negative", '
                    '"reasoning": "Related entity but does not satisfy the query."}'
                )
            }
        ]


def test_single_doc_hidden_doc_id_backfills_input_doc_id():
    annotator = BatchCandidateAnnotator(task_name="dbpedia", llm_cfg={})
    annotator.evaluation_instruction = "Classify candidate documents."
    candidate = {
        "doc_id": "<dbpedia:American_Chinese_cuisine>",
        "doc": "Title: American Chinese cuisine\nText: A cuisine style.",
    }

    with patch("Batch_Candidate_Annotator.Assistant", FakeAssistant):
        result = annotator._annotate_one(
            "Chinese food",
            [candidate],
            candidates_num_each_anno=1,
            hide_doc_id_for_single_doc=True,
        )

    assert result["annotations"] == [
        {
            "doc_id": "<dbpedia:American_Chinese_cuisine>",
            "score": 1,
            "classification": "hard_negative",
            "reasoning": "Related entity but does not satisfy the query.",
        }
    ]
    assert result["hard_negative_doc_ids"] == ["<dbpedia:American_Chinese_cuisine>"]
    assert "<dbpedia:American_Chinese_cuisine>" not in FakeAssistant.last_messages[0]["content"]


class FakeMultiDocAssistant:
    last_messages = None

    def __init__(self, llm=None, system_message=None):
        self.llm = llm
        self.system_message = system_message

    def run(self, messages):
        FakeMultiDocAssistant.last_messages = messages
        yield [
            {
                "content": (
                    '{"annotations": ['
                    '{"doc_id": "0", "score": 3, "classification": "positive", "reasoning": "Directly answers."},'
                    '{"doc_id": "1", "score": 1, "classification": "hard_negative", "reasoning": "Related but insufficient."}'
                    ']}'
                )
            }
        ]


def test_multi_doc_annotation_uses_local_doc_ids_and_restores_original_ids():
    annotator = BatchCandidateAnnotator(task_name="dbpedia", llm_cfg={})
    annotator.evaluation_instruction = "Classify candidate documents and echo doc_id."
    candidates = [
        {
            "doc_id": "<dbpedia:American_Chinese_cuisine>",
            "doc": "Title: American Chinese cuisine\nText: A cuisine style.",
        },
        {
            "doc_id": "msmarco-passage-00123:abc/xyz",
            "doc": "Title: Nearby topic\nText: Related but incomplete.",
        },
    ]

    with patch("Batch_Candidate_Annotator.Assistant", FakeMultiDocAssistant):
        result = annotator._annotate_one("Chinese food", candidates)

    assert result["annotations"] == [
        {
            "doc_id": "<dbpedia:American_Chinese_cuisine>",
            "score": 3,
            "classification": "positive",
            "reasoning": "Directly answers.",
        },
        {
            "doc_id": "msmarco-passage-00123:abc/xyz",
            "score": 1,
            "classification": "hard_negative",
            "reasoning": "Related but insufficient.",
        },
    ]
    assert result["positive_doc_ids"] == ["<dbpedia:American_Chinese_cuisine>"]
    assert result["hard_negative_doc_ids"] == ["msmarco-passage-00123:abc/xyz"]
    prompt = FakeMultiDocAssistant.last_messages[0]["content"]
    assert "[doc_id: 0]" in prompt
    assert "[doc_id: 1]" in prompt
    assert "<dbpedia:American_Chinese_cuisine>" not in prompt
    assert "msmarco-passage-00123:abc/xyz" not in prompt


if __name__ == "__main__":
    test_single_doc_hidden_doc_id_backfills_input_doc_id()
    test_multi_doc_annotation_uses_local_doc_ids_and_restores_original_ids()
