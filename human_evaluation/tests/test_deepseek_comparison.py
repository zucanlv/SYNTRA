import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_deepseek import Target, extract_deepseek_annotations, named_metrics


class DeepSeekComparisonTest(unittest.TestCase):
    def test_streaming_alignment_prefers_query_doc_id_and_document(self):
        payload = {
            "results": [
                {
                    "queries": [
                        {
                            "query": "q",
                            "candidates": [
                                {"doc_id": "d1", "doc": "document"},
                                {"doc_id": "d2", "doc": "other"},
                            ],
                            "annotations": [
                                {
                                    "doc_id": "d1",
                                    "score": 2,
                                    "classification": "positive",
                                    "reasoning": "useful",
                                },
                                {
                                    "doc_id": "d2",
                                    "score": 0,
                                    "classification": "easy negative",
                                    "reasoning": "irrelevant",
                                },
                            ],
                        }
                    ]
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotated.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result, diagnostics = extract_deepseek_annotations(
                path,
                [Target("sample-1", "dataset", "q", "document", "d1")],
            )
        self.assertEqual(result["sample-1"].score, 2)
        self.assertEqual(result["sample-1"].classification, "positive")
        self.assertEqual(result["sample-1"].match_method, "query_doc_id_document")
        self.assertEqual(diagnostics["aligned_count"], 1)

    def test_alignment_rejects_missing_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotated.json"
            path.write_text(json.dumps({"results": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing=\\['sample-1'\\]"):
                extract_deepseek_annotations(
                    path,
                    [Target("sample-1", "dataset", "q", "document", "d1")],
                )

    def test_named_metrics_make_confusion_orientation_explicit(self):
        metrics = named_metrics(
            ["positive", "negative"],
            ["positive", "positive"],
            ("positive", "negative"),
            "qwen",
            "deepseek",
        )
        self.assertEqual(metrics["matches"], 1)
        self.assertEqual(metrics["confusion_matrix_rows"], "qwen")
        self.assertEqual(metrics["confusion_matrix_columns"], "deepseek")
        self.assertEqual(metrics["left_distribution"], {"positive": 1, "negative": 1})


if __name__ == "__main__":
    unittest.main()
