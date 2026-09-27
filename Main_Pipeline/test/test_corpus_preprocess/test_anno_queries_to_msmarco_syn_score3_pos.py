import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "corpus_preprocess"
    / "anno_queries_to_msmarco_syn_score3_pos.py"
)


class Score3PositiveConversionTest(unittest.TestCase):
    def test_uses_only_score_3_as_positive_and_preserves_hard_negatives(self):
        self.assertTrue(SCRIPT_PATH.is_file(), f"missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("score3_converter", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        result = {
            "queries": [
                {
                    "query": "  implement a Pony function  ",
                    "candidates": [
                        {"doc_id": "core", "doc": "core Pony API"},
                        {"doc_id": "support", "doc": "basic loop syntax"},
                        {"doc_id": "hard", "doc": "plausible but wrong API"},
                    ],
                    "annotations": [
                        {"doc_id": "core", "score": 3, "classification": "positive"},
                        {"doc_id": "support", "score": 2, "classification": "positive"},
                        {"doc_id": "hard", "score": 1, "classification": "hard negative"},
                    ],
                    "positive_doc_ids": ["core", "support"],
                    "hard_negative_doc_ids": ["hard", "missing"],
                }
            ]
        }

        record = module.result_to_record(result, "retrieve Pony docs")

        self.assertEqual(
            record,
            {
                "prompt": "retrieve Pony docs",
                "query": "implement a Pony function",
                "pos": ["core Pony API"],
                "neg": ["plausible but wrong API"],
            },
        )

    def test_skips_record_without_score_3_positive(self):
        self.assertTrue(SCRIPT_PATH.is_file(), f"missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("score3_converter_no_pos", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = {
            "queries": [
                {
                    "query": "task",
                    "candidates": [
                        {"doc_id": "support", "doc": "basic syntax"},
                        {"doc_id": "hard", "doc": "wrong API"},
                    ],
                    "annotations": [{"doc_id": "support", "score": 2}],
                    "hard_negative_doc_ids": ["hard"],
                }
            ]
        }

        self.assertIsNone(module.result_to_record(result, "prompt"))


if __name__ == "__main__":
    unittest.main()
