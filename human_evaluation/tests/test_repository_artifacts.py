import hashlib
import json
import re
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryArtifactsTest(unittest.TestCase):
    def test_public_samples_have_exact_blinded_schema_and_dataset_counts(self):
        samples = json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))
        self.assertEqual(len(samples), 150)
        self.assertEqual(
            Counter(sample["dataset"] for sample in samples),
            {"msmarco": 50, "text2sql": 50, "theoremqa-theorems": 50},
        )
        self.assertEqual(len({sample["sample_id"] for sample in samples}), 150)
        for sample in samples:
            self.assertEqual(
                set(sample), {"sample_id", "dataset", "query", "document"}
            )

    def test_guidelines_match_the_logged_instruction_versions(self):
        expected = {
            "msmarco.md": "fcf3323646354e9926a8c459de8d6817c8e8e9f8e8a8ed6c0d887736194c3e71",
            "text2sql.md": "2d1fdc97bd79d626942c059a341a1f0217116f0406bdc95f2b8a62ca90555982",
            "theoremqa-theorems.md": "b4a8614edf4dd3b7294470385ee1a4d06d67b94c43e17c07163354dd6b3e283b",
        }
        for filename, digest in expected.items():
            with self.subTest(filename=filename):
                content = (ROOT / "guidelines" / filename).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)

    def test_every_katex_font_reference_is_bundled(self):
        vendor = ROOT / "web" / "vendor"
        css = (vendor / "katex.min.css").read_text(encoding="utf-8")
        references = re.findall(r"url\(([^)]+)\)", css)
        self.assertTrue(references)
        for reference in references:
            with self.subTest(reference=reference):
                self.assertTrue((vendor / reference.strip('"\'')).is_file())


if __name__ == "__main__":
    unittest.main()
