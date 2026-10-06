"""Startup regression; requires the synthesis environment, but no model or API."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PIPELINE = Path(__file__).resolve().parents[1] / "Main_Pipeline"


class PipelineStartup(unittest.TestCase):
    def test_help_in_empty_directory_needs_no_log_directories(self):
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "CUDA_VISIBLE_DEVICES": ""}
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENAI_BASE_URL", None)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(PIPELINE / "main.py"), "--help"],
                cwd=directory, env=env, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("--config", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
