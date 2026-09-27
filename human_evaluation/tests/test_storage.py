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


class AnnotationStoreTest(unittest.TestCase):
    def test_save_creates_csv_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "annotator_1.csv"
            store = AnnotationStore(SAMPLES, output, "annotator_1")

            store.save("msmarco-001", 2)

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                rows,
                [
                    {
                        "annotator_id": "annotator_1",
                        "sample_id": "msmarco-001",
                        "dataset": "msmarco",
                        "score": "2",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
