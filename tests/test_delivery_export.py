"""Exercise the documented export commands using temporary synthetic records."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PREPROCESS = Path(__file__).resolve().parents[1] / "Main_Pipeline/corpus_preprocess"


class DeliveryExport(unittest.TestCase):
    def test_export_and_origin_positive_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A shallow checkout must not require five ancestors to start.
            converter = root / "convert.py"
            shutil.copyfile(PREPROCESS / "anno_queries_to_msmarco_syn.py", converter)
            annotated = root / "annotated.json"
            candidates = [{"doc_id": "p1", "doc": "origin passage"},
                          {"doc_id": "p2", "doc": "another positive"},
                          {"doc_id": "n1", "doc": "hard negative"}]
            annotated.write_text(json.dumps({"results": [
                {"queries": [{"query": "kept query", "candidates": candidates,
                              "positive_doc_ids": ["p1", "p2"],
                              "hard_negative_doc_ids": ["n1"]}]},
                {"queries": [{"query": "no negative", "candidates": candidates,
                              "positive_doc_ids": ["p1"],
                              "hard_negative_doc_ids": []}]},
            ]}))
            exported = root / "train.jsonl"
            self.run_command(converter, "--input", annotated, "--output", exported,
                             "--prompt", "Retrieve a passage.")
            rows = [json.loads(line) for line in exported.read_text().splitlines()]
            self.assertEqual(rows, [{"prompt": "Retrieve a passage.", "query": "kept query",
                                     "pos": ["origin passage", "another positive"],
                                     "neg": ["hard negative"]}])
            diverse = root / "diverse.json"
            diverse.write_text(json.dumps({"results": [
                {"doc": "origin passage", "queries": [{"query": "kept query"}]}
            ]}))
            filtered = root / "filtered.jsonl"
            self.run_command(PREPROCESS / "4-29-exclude-other-pos-save-non-pos.py",
                             "--input-jsonl", exported, "--diverse-query", diverse,
                             "--output", filtered)
            rows[0]["pos"] = ["origin passage"]
            self.assertEqual([json.loads(line) for line in filtered.read_text().splitlines()], rows)

    def run_command(self, *args):
        result = subprocess.run([sys.executable, *map(str, args)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
