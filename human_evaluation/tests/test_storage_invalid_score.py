import tempfile
import unittest
from pathlib import Path

from annotation_app.storage import AnnotationStore


class AnnotationStoreInvalidScoreTest(unittest.TestCase):
    def test_invalid_score_is_rejected_without_writing(self):
        samples = [
            {
                "sample_id": "msmarco-001",
                "dataset": "msmarco",
                "query": "q",
                "document": "d",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "annotator_1.csv"
            store = AnnotationStore(samples, output, "annotator_1")

            with self.assertRaisesRegex(ValueError, "score"):
                store.save("msmarco-001", 4)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
