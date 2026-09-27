import csv
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from annotation_app.storage import (
    AnnotationDataError,
    AnnotationStore,
    load_samples,
    validate_annotator_id,
)


def make_samples(count=4):
    return [
        {
            "sample_id": f"msmarco-{index:03d}",
            "dataset": "msmarco",
            "query": f"query {index}",
            "document": f"document {index}",
        }
        for index in range(count)
    ]


class AnnotationStoreSafetyTest(unittest.TestCase):
    def test_annotator_id_blocks_paths_and_accepts_simple_ids(self):
        self.assertEqual(validate_annotator_id("annotator_1"), "annotator_1")
        for invalid in ("", "../escape", "a/b", "name.csv", "标注者1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "annotator"):
                    validate_annotator_id(invalid)

    def test_unknown_sample_is_rejected_as_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(
                make_samples(1), Path(tmp) / "annotator_1.csv", "annotator_1"
            )
            with self.assertRaisesRegex(ValueError, "unknown sample"):
                store.save("missing", 2)

    def test_corrupt_existing_csv_stops_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "annotator_1.csv"
            with output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("annotator_id", "sample_id", "dataset", "score"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "annotator_id": "someone_else",
                        "sample_id": "msmarco-000",
                        "dataset": "msmarco",
                        "score": 2,
                    }
                )
            with self.assertRaisesRegex(AnnotationDataError, "annotator_id"):
                AnnotationStore(make_samples(1), output, "annotator_1")

    def test_concurrent_saves_keep_every_unique_row(self):
        samples = make_samples(20)
        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(
                samples, Path(tmp) / "annotator_1.csv", "annotator_1"
            )
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(
                    executor.map(
                        lambda sample: store.save(sample["sample_id"], 2), samples
                    )
                )
            self.assertEqual(len(store.scores()), 20)

    def test_load_samples_rejects_answer_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.json"
            payload = make_samples(1)
            payload[0]["llm_score"] = 3
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AnnotationDataError, "fields"):
                load_samples(path)


if __name__ == "__main__":
    unittest.main()
