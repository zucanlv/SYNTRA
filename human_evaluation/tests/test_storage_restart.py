import csv
import tempfile
import unittest
from pathlib import Path

from annotation_app.storage import AnnotationStore


SAMPLES = [
    {
        "sample_id": "msmarco-001",
        "dataset": "msmarco",
        "query": "example query",
        "document": "example document",
    }
]


class AnnotationStoreRestartTest(unittest.TestCase):
    def test_restart_restores_progress_and_update_replaces_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "annotator_1.csv"
            AnnotationStore(SAMPLES, output, "annotator_1").save("msmarco-001", 1)

            restarted = AnnotationStore(SAMPLES, output, "annotator_1")
            self.assertEqual(restarted.scores(), {"msmarco-001": 1})

            restarted.save("msmarco-001", 3)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["score"], "3")


if __name__ == "__main__":
    unittest.main()
