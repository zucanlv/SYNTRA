import json
import tempfile
import unittest
from pathlib import Path

from annotation_app.translations import TranslationDataError, load_translations


class TranslationDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "translations.json"
        self.samples = [
            {"sample_id": "sample-1", "dataset": "demo", "query": "q", "document": "d"}
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def test_loads_complete_translation_map(self):
        self.path.write_text(
            json.dumps({"sample-1": {"query": "查询", "document": "文档"}}),
            encoding="utf-8",
        )
        translations = load_translations(self.path, self.samples)
        self.assertEqual(translations["sample-1"]["query"], "查询")

    def test_rejects_missing_sample_translation(self):
        self.path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(TranslationDataError, "missing sample IDs"):
            load_translations(self.path, self.samples)

    def test_rejects_extra_translation_fields(self):
        self.path.write_text(
            json.dumps(
                {"sample-1": {"query": "查询", "document": "文档", "score": "3"}}
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(TranslationDataError, "exactly"):
            load_translations(self.path, self.samples)


if __name__ == "__main__":
    unittest.main()
