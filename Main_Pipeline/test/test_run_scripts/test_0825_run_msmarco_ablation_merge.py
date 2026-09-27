import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "run-scripts/0825-run-msmarco-ablation-merge.sh"


class MsmarcoAblationMergeScriptTest(unittest.TestCase):
    def test_dry_run_plans_four_merged_training_and_dual_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "MODEL_ROOT": str(output_root / "models"),
                    "LOG_ROOT": str(output_root / "logs"),
                    "MSMARCO_EVAL_ROOT": str(output_root / "msmarco"),
                    "NANOBEIR_EVAL_ROOT": str(output_root / "nanobeir"),
                }
            )

            result = subprocess.run(
                ["bash", str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("[DRY RUN] train "), 4)
            self.assertEqual(result.stdout.count("[DRY RUN] eval-msmarco "), 4)
            self.assertEqual(result.stdout.count("[DRY RUN] eval-nanobeir "), 4)

            expected_stems = (
                "without-query-gen-instr-10k",
                "without-adattr-10k",
                "hn-mine-10k",
                "hn-mine-10k-ori-pos-anno",
            )
            for stem in expected_stems:
                self.assertIn(f"{stem}.jsonl", result.stdout)
                self.assertIn(
                    f"{stem}_exclude_other_pos_save_non_pos.jsonl",
                    result.stdout,
                )

            self.assertFalse((output_root / "models").exists())
            self.assertFalse((output_root / "logs").exists())
            self.assertFalse((output_root / "msmarco").exists())
            self.assertFalse((output_root / "nanobeir").exists())


if __name__ == "__main__":
    unittest.main()
