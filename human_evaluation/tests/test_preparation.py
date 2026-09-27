import csv
import json
import tempfile
import unittest
from pathlib import Path

from annotation_app.preparation import DatasetSource, prepare_experiment


def write_source(path: Path, count: int) -> None:
    candidates = []
    annotations = []
    for index in range(count):
        candidates.append(
            {
                "doc_id": f"d{index}",
                "doc": f"document {index}",
                "score": 0.5,
                "rank": index + 1,
                "is_original": index == 0,
            }
        )
        annotations.append(
            {
                "doc_id": f"d{index}",
                "score": index % 4,
                "classification": "positive",
                "reasoning": f"reason {index}",
            }
        )
    payload = {
        "results": [
            {
                "queries": [
                    {
                        "query": "test query",
                        "candidates": candidates,
                        "annotations": annotations,
                    }
                ]
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class PreparationTest(unittest.TestCase):
    def test_prepare_writes_blinded_public_data_and_private_audit_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            write_source(source, 4)

            summary = prepare_experiment(
                [DatasetSource("msmarco", source)],
                sample_count=2,
                seed=123,
                public_path=root / "data" / "samples.json",
                private_dir=root / "private",
            )

            public = json.loads(
                (root / "data" / "samples.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(public), 2)
            self.assertEqual(
                set(public[0]), {"sample_id", "dataset", "query", "document"}
            )
            with (root / "private" / "answer_key.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                private_rows = list(csv.DictReader(handle))
            self.assertEqual(len(private_rows), 2)
            self.assertIn("llm_score", private_rows[0])
            manifest = json.loads(
                (root / "private" / "sampling_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["seed"], 123)
            self.assertEqual(manifest["datasets"][0]["valid_pair_count"], 4)
            self.assertEqual(len(manifest["datasets"][0]["source_sha256"]), 64)
            self.assertEqual(summary["public_sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
