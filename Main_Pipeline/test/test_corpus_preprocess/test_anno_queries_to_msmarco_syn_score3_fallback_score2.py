import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "corpus_preprocess"
    / "anno_queries_to_msmarco_syn_score3_fallback_score2.py"
)


def _load_converter():
    if not SCRIPT_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location("score3_fallback_converter", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(annotations):
    return {
        "queries": [
            {
                "query": "  implement a Pony function  ",
                "candidates": [
                    {"doc_id": "core", "doc": "core Pony API"},
                    {"doc_id": "support-a", "doc": "supporting syntax A"},
                    {"doc_id": "support-b", "doc": "supporting syntax B"},
                    {"doc_id": "hard", "doc": "plausible but wrong API"},
                ],
                "annotations": annotations,
                "hard_negative_doc_ids": ["hard"],
            }
        ]
    }


class Score3FallbackScore2ConversionTest(unittest.TestCase):
    def test_prefers_score_3_without_mixing_score_2(self):
        module = _load_converter()
        self.assertIsNotNone(module, f"missing script: {SCRIPT_PATH}")
        result = _result(
            [
                {"doc_id": "support-a", "score": 2},
                {"doc_id": "core", "score": 3},
                {"doc_id": "support-b", "score": 2},
            ]
        )

        record = module.result_to_record(result, "retrieve Pony docs")

        self.assertEqual(record["pos"], ["core Pony API"])
        self.assertEqual(record["neg"], ["plausible but wrong API"])

    def test_uses_all_score_2_when_no_score_3_is_resolvable(self):
        module = _load_converter()
        self.assertIsNotNone(module, f"missing script: {SCRIPT_PATH}")
        result = _result(
            [
                {"doc_id": "support-a", "score": 2},
                {"doc_id": "missing-core", "score": 3},
                {"doc_id": "support-b", "score": 2},
            ]
        )

        record = module.result_to_record(result, "retrieve Pony docs")

        self.assertEqual(
            record["pos"],
            ["supporting syntax A", "supporting syntax B"],
        )
        self.assertEqual(record["query"], "implement a Pony function")


if __name__ == "__main__":
    unittest.main()
