import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_delivery", ROOT / "scripts/check_delivery.py")
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


class DeliveryChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.corpus = self.root / "corpus.jsonl"
        self.corpus.write_text('{"id":"one","text":"example"}\n')
        self.cfg = dict(task="msmarco", corpus_path=str(self.corpus),
                        faiss_dir=str(self.root / "index"),
                        instructions_store_dir=str(self.root / "instructions"),
                        fully_annotation=True, skip_index=False, test_mode=False,
                        llm_model="test-model")
        self.env = dict(OPENAI_API_KEY="test-secret", OPENAI_BASE_URL="http://localhost/v1")

    def test_refine_is_read_only(self):
        before = set(self.root.rglob("*"))
        self.assertEqual(delivery.check_config(self.cfg, "refine", self.env), [])
        self.assertEqual(set(self.root.rglob("*")), before)

    def test_credentials_are_not_echoed(self):
        self.cfg["llm_api_key"] = "sensitive-test-value"
        errors = delivery.check_config(self.cfg, "refine", self.env)
        self.assertTrue(errors)
        self.assertNotIn("sensitive-test-value", str(errors))

    def test_wrong_task_and_missing_corpus(self):
        self.cfg.update(task="unregistered", corpus_path="missing")
        self.assertGreaterEqual(len(delivery.check_config(self.cfg, "refine", self.env)), 2)

    def test_resume_inputs_rejected_for_this_route(self):
        self.cfg["step5_query2passage_input"] = "/some/intermediate.json"
        self.assertTrue(delivery.check_config(self.cfg, "refine", self.env))

    def test_production_needs_final_nonempty_instructions(self):
        store = Path(self.cfg["instructions_store_dir"])
        store.mkdir()
        for prefix in ("IdAttrRefine", "QueryInstrRefine", "AnnoInstrRefine"):
            (store / f"{prefix}_msmarco_attempt1_test.txt").write_text("intermediate")
        self.assertEqual(len(delivery.check_config(self.cfg, "production", self.env)), 3)
        for prefix in ("IdAttrRefine", "QueryInstrRefine", "AnnoInstrRefine"):
            (store / f"{prefix}_msmarco_20260101.txt").write_text("final")
        self.assertEqual(delivery.check_config(self.cfg, "production", self.env), [])
        self.cfg["test_mode"] = True
        self.assertTrue(delivery.check_config(self.cfg, "production", self.env))

    def test_placeholder_lab_path_and_unexpanded_variable(self):
        self.cfg.update(llm_model="REPLACE_MODEL", faiss_dir="/data/share/project/index",
                        instructions_store_dir="${OUTPUT}/instructions")
        self.assertGreaterEqual(len(delivery.check_config(self.cfg, "refine", self.env)), 3)

    def test_malformed_yaml_cli_does_not_echo_secrets(self):
        config = self.root / "bad.yaml"
        config.write_text('llm_api_key: [sensitive-test-value')
        p = subprocess.run([sys.executable, str(ROOT / "scripts/check_delivery.py"),
                            "--config", str(config), "--phase", "refine"],
                           cwd=ROOT / "Main_Pipeline", capture_output=True, text=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertNotEqual(p.returncode, 0)
        self.assertNotIn("sensitive-test-value", p.stdout + p.stderr)


if __name__ == "__main__":
    unittest.main()
