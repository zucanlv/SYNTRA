import json
import tempfile
import unittest
from pathlib import Path

from annotation_app.sampling import (
    blind_samples,
    iter_source_pairs,
    reservoir_sample,
)


class StubRandom:
    def __init__(self, values):
        self.values = iter(values)

    def randrange(self, stop):
        value = next(self.values)
        if not 0 <= value < stop:
            raise AssertionError(f"stub value {value} outside randrange({stop})")
        return value


class SamplingTest(unittest.TestCase):
    def test_reservoir_samples_from_the_full_item_stream(self):
        selected, seen = reservoir_sample(
            iter(["a", "b", "c", "d", "e"]), 2, StubRandom([1, 3, 0])
        )
        self.assertEqual(selected, ["e", "c"])
        self.assertEqual(seen, 5)

    def test_source_pair_joins_annotation_to_candidate(self):
        payload = {
            "results": [
                {
                    "queries": [
                        {
                            "query": "When can I move in?",
                            "candidates": [
                                {
                                    "doc_id": "d1",
                                    "doc": "Move-in starts Monday.",
                                    "score": 0.75,
                                    "rank": 1,
                                    "is_original": True,
                                }
                            ],
                            "annotations": [
                                {
                                    "doc_id": "d1",
                                    "score": 3,
                                    "classification": "positive",
                                    "reasoning": "Direct answer.",
                                }
                            ],
                        }
                    ]
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            pairs = list(iter_source_pairs(path, "msmarco"))

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["query"], "When can I move in?")
        self.assertEqual(pairs[0]["document"], "Move-in starts Monday.")
        self.assertEqual(pairs[0]["source_doc_id"], "d1")
        self.assertEqual(pairs[0]["llm_score"], 3)
        self.assertEqual(pairs[0]["retrieval_rank"], 1)

    def test_blinding_exposes_only_four_public_fields(self):
        selected = {
            "msmarco": [
                {
                    "query": "q",
                    "document": "d",
                    "source_doc_id": "secret",
                    "llm_score": 3,
                    "llm_classification": "positive",
                    "llm_reasoning": "secret reasoning",
                    "retrieval_score": 0.9,
                    "retrieval_rank": 1,
                    "is_original": True,
                    "source_result_index": 0,
                    "source_query_index": 0,
                }
            ]
        }

        public, private = blind_samples(selected)

        self.assertEqual(
            public,
            [
                {
                    "sample_id": "msmarco-001",
                    "dataset": "msmarco",
                    "query": "q",
                    "document": "d",
                }
            ],
        )
        self.assertEqual(private[0]["sample_id"], "msmarco-001")
        self.assertEqual(private[0]["llm_score"], 3)
        self.assertNotIn("llm_score", public[0])


if __name__ == "__main__":
    unittest.main()
