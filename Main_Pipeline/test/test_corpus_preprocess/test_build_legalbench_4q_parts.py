import os
import pathlib
import subprocess
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "run-scripts" / "0813-build-legalbench-rag-4q-training-data.sh"


class BuildLegalbench4qPartsTest(unittest.TestCase):
    def test_dry_run_merges_three_parts_and_builds_four_positions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            parts = []
            for index in range(1, 4):
                part = tmp_path / f"annotated-part{index}.json"
                part.write_text('{"results": []}\n', encoding="utf-8")
                parts.append(part)
            diverse_query = tmp_path / "diverse-query.json"
            diverse_query.write_text('{"results": []}\n', encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "ANNOTATED_PART1": str(parts[0]),
                    "ANNOTATED_PART2A": str(parts[1]),
                    "ANNOTATED_PART2B": str(parts[2]),
                    "DIVERSE_QUERY_FILE": str(diverse_query),
                    "MERGED_ANNOTATED_FILE": str(tmp_path / "merged.json"),
                    "OUTPUT_DIR": str(tmp_path / "output"),
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
            self.assertEqual(output.count("merge_pipeline_json.py"), 1, output)
            for part in parts:
                self.assertIn(str(part), output)
            for query_position in range(1, 5):
                self.assertEqual(
                    output.count(f"--query-position {query_position}"), 2, output
                )
                self.assertIn(
                    f"legalbench-rag-4q-query{query_position}-20260813.jsonl", output
                )
                self.assertIn(
                    f"legalbench-rag-4q-query{query_position}-20260813"
                    "_exclude_other_pos_save_non_pos.jsonl",
                    output,
                )
            self.assertIn("validated 4 query-position builds", output)


if __name__ == "__main__":
    unittest.main()
