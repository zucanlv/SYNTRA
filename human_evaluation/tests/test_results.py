import csv
import tempfile
import unittest
from pathlib import Path

from annotation_app.results import ResultValidationError, merge_annotations
from annotation_app.storage import AnnotationStore


SAMPLES = [
    {"sample_id": "msmarco-001", "dataset": "msmarco", "query": "q1", "document": "d1"},
    {"sample_id": "text2sql-001", "dataset": "text2sql", "query": "q2", "document": "d2"},
]


def write_submission(directory: Path, annotator: str, scores: list[int]) -> Path:
    path = directory / f"{annotator}.csv"
    store = AnnotationStore(SAMPLES, path, annotator)
    for sample, score in zip(SAMPLES, scores):
        store.save(sample["sample_id"], score)
    return path


class ResultsTest(unittest.TestCase):
    def test_merge_preserves_three_raw_scores_without_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = [
                write_submission(root, "annotator_1", [0, 1]),
                write_submission(root, "annotator_2", [1, 2]),
                write_submission(root, "annotator_3", [2, 3]),
            ]
            output = root / "merged.csv"

            summary = merge_annotations(inputs, SAMPLES, output, expected_count=3)

            with output.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(
                reader.fieldnames,
                ["sample_id", "dataset", "annotator_1", "annotator_2", "annotator_3"],
            )
            self.assertEqual(rows[0]["annotator_3"], "2")
            self.assertNotIn("ground_truth", rows[0])
            self.assertEqual(summary["sample_count"], 2)

    def test_merge_rejects_an_incomplete_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = [
                write_submission(root, "annotator_1", [0]),
                write_submission(root, "annotator_2", [1, 2]),
                write_submission(root, "annotator_3", [2, 3]),
            ]
            with self.assertRaisesRegex(ResultValidationError, "incomplete"):
                merge_annotations(inputs, SAMPLES, root / "merged.csv", expected_count=3)


if __name__ == "__main__":
    unittest.main()
