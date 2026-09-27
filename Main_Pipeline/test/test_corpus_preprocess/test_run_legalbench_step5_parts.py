import os
import pathlib
import subprocess
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    PROJECT_ROOT / "run-scripts" / "0812-run-legalbench-rag-4q-step5-parts.sh"
)


class RunLegalbenchStep5PartsTest(unittest.TestCase):
    def test_dry_run_invokes_step5_for_all_three_parts_in_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            results_dir = tmp_path / "legalbench-rag_20260811_202254"
            results_dir.mkdir()
            part_names = (
                "Query2Passage_Main_20260811_212858_part1.json",
                "Query2Passage_Main_20260811_212858_part2a.json",
                "Query2Passage_Main_20260811_212858_part2b.json",
            )
            for part_name in part_names:
                (results_dir / part_name).write_text('{"results": []}\n', encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "LEGALBENCH_RESULTS_DIR": str(results_dir),
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
            self.assertEqual(output.count("--step5-query2passage-input"), 3, output)
            positions = [output.index(part_name) for part_name in part_names]
            self.assertEqual(positions, sorted(positions), output)
            self.assertIn("DRY_RUN complete: validated 3 Step 5 runs", output)


if __name__ == "__main__":
    unittest.main()
