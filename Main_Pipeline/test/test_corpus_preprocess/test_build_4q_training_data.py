import os
import pathlib
import subprocess
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "run-scripts" / "0812-build-4q-generalization-training-data.sh"


class Build4qTrainingDataTest(unittest.TestCase):
    def test_dry_run_builds_raw_and_filtered_jsonl_for_each_query_position(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            results_root = tmp_path / "results"
            code_output_dir = tmp_path / "code-output"
            domain_output_root = tmp_path / "domain-output"

            fixtures = {
                "codetrans-dl_20260811_202056": (
                    "Annotated_Main_20260811_205419.json",
                    "DiverseQuery_Main_20260811_205055.json",
                ),
                "legalbench-rag_20260811_202254": (
                    "Annotated_Main_20260811_214453.json",
                    "DiverseQuery_Main_20260811_212414.json",
                ),
                "medical_qa_20260811_202419": (
                    "Annotated_Main_20260811_212858.json",
                    "DiverseQuery_Main_20260811_212108.json",
                ),
            }
            for result_dir, filenames in fixtures.items():
                directory = results_root / result_dir
                directory.mkdir(parents=True)
                for filename in filenames:
                    (directory / filename).write_text('{"results": []}\n', encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "RESULTS_ROOT": str(results_root),
                    "CODE_OUTPUT_DIR": str(code_output_dir),
                    "DOMAIN_OUTPUT_ROOT": str(domain_output_root),
                    "DRY_RUN": "1",
                    "SKIP_CONDA_ACTIVATE": "1",
                }
            )
            completed = subprocess.run(
                ["bash", str(SCRIPT_PATH)],
                cwd=PROJECT_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = completed.stdout
            for query_position in range(1, 5):
                self.assertEqual(
                    output.count(f"--query-position {query_position}"),
                    6,
                    output,
                )

            expected_stems = (
                "codetrans-dl-4q",
                "legalbench-rag-4q",
                "medical_qa-all-4q",
            )
            for stem in expected_stems:
                for query_position in range(1, 5):
                    raw_name = f"{stem}-query{query_position}-20260812.jsonl"
                    filtered_name = (
                        f"{stem}-query{query_position}-20260812"
                        "_exclude_other_pos_save_non_pos.jsonl"
                    )
                    self.assertIn(f"output       :", output)
                    self.assertIn(raw_name, output)
                    self.assertIn(filtered_name, output)

            self.assertIn("validated 12 query-position builds", output)


if __name__ == "__main__":
    unittest.main()
