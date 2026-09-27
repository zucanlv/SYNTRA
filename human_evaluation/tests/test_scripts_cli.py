import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScriptsCliTest(unittest.TestCase):
    def test_admin_scripts_expose_help_without_third_party_runtime_imports(self):
        for name in (
            "prepare_samples.py",
            "validate_annotations.py",
            "merge_annotations.py",
        ):
            with self.subTest(script=name):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / name), "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
