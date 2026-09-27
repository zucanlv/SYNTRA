import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.scirepeval_full_corpus import (  # noqa: E402
    flagembedding_model_options,
    load_embedding_texts,
)


class SciRepEvalFullCorpusTest(unittest.TestCase):
    def test_flagembedding_options_match_training_encoder(self):
        options = flagembedding_model_options(device="cuda:2", batch_size=7)

        self.assertEqual(options["model_class"], "decoder-only-base")
        self.assertEqual(options["pooling_method"], "last_token")
        self.assertTrue(options["normalize_embeddings"])
        self.assertFalse(options["use_fp16"])
        self.assertIsNone(options["query_instruction_for_retrieval"])
        self.assertEqual(options["query_instruction_format"], "Instruct: {}\nQuery: {}")
        self.assertEqual(options["devices"], "cuda:2")
        self.assertEqual(options["batch_size"], 7)
        self.assertEqual(options["query_max_length"], 512)
        self.assertEqual(options["passage_max_length"], 512)

    def test_load_embedding_texts_matches_official_field_joining(self):
        rows = [
            {
                "doc_id": "q1",
                "query": "scientific query",
                "candidates": [
                    {
                        "doc_id": "d1",
                        "title": "Paper title",
                        "abstract": "Paper abstract",
                        "venue": "",
                        "year": 2024,
                    }
                ],
            },
            {
                "doc_id": "q2",
                "query": "another query",
                "candidates": [
                    {
                        "doc_id": "d1",
                        "title": "duplicate must not replace first text",
                        "abstract": "ignored",
                        "venue": "ignored",
                        "year": 2025,
                    },
                    {
                        "doc_id": "d2",
                        "title": "Second paper",
                        "abstract": None,
                        "venue": "Venue",
                        "year": None,
                    },
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = Path(temp_dir) / "evaluation.jsonl"
            metadata.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            queries, corpus = load_embedding_texts(str(metadata), "<eos>")

        self.assertEqual(
            queries,
            {"q1": "scientific query", "q2": "another query"},
        )
        self.assertEqual(
            corpus,
            {
                "d1": "Paper title <eos> Paper abstract <eos> 2024",
                "d2": "Second paper <eos> Venue",
            },
        )


if __name__ == "__main__":
    unittest.main()
