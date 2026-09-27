import json
import pathlib
import subprocess
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "Main_Pipeline" / "corpus_preprocess" / "merge_pipeline_json.py"


class MergePipelineJsonTest(unittest.TestCase):
    def test_merges_results_in_input_order_and_preserves_shared_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            inputs = []
            for index, doc_ids in enumerate((("a", "b"), ("c",), ("d", "e")), 1):
                path = tmp_path / f"part{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "input_path": "source.jsonl",
                            "num_queries": 4,
                            "use_idattr": True,
                            "results": [{"doc_id": doc_id} for doc_id in doc_ids],
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append(path)

            output = tmp_path / "merged.json"
            completed = subprocess.run(
                ["python", str(SCRIPT_PATH), "--output", str(output), *map(str, inputs)],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            merged = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(merged["input_path"], "source.jsonl")
            self.assertEqual(merged["num_queries"], 4)
            self.assertIs(merged["use_idattr"], True)
            self.assertEqual(
                [item["doc_id"] for item in merged["results"]],
                ["a", "b", "c", "d", "e"],
            )
            self.assertIn('"total_results": 5', completed.stdout)


if __name__ == "__main__":
    unittest.main()
