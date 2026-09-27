import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "Main_Pipeline/corpus_preprocess/summarize_mteb_results.py"


class SummarizeMtebResultsTest(unittest.TestCase):
    def test_summarizes_task_files_without_benchmark_registry_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            fixtures = {
                "NanoTaskA.json": {
                    "task_name": "NanoTaskA",
                    "scores": {"test": [{"main_score": 0.5}]},
                },
                "NanoTaskB.json": {
                    "task_name": "NanoTaskB",
                    "scores": {"train": [{"main_score": 0.7}]},
                },
                "model_meta.json": {"model_name": "example"},
            }
            for name, payload in fixtures.items():
                (result_dir / name).write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            result = subprocess.run(
                ["python", str(SCRIPT_PATH), str(result_dir)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("final score 2 60.0", result.stdout)
            self.assertIn("NanoTaskA 50.0", result.stdout)
            self.assertIn("NanoTaskB 70.0", result.stdout)
            self.assertNotIn("model_meta", result.stdout)


if __name__ == "__main__":
    unittest.main()
