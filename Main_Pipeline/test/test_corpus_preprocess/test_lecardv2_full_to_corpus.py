import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "corpus_preprocess"
    / "lecardv2_full_to_corpus.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("lecardv2_full_to_corpus", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LeCaRDv2FullCorpusTest(unittest.TestCase):
    def test_conversion_matches_mteb_schema_and_text_construction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "candidate_55192"
            source.mkdir()
            (source / "10.json").write_text(
                json.dumps(
                    {
                        "pid": 10,
                        "qw": "判决书乙",
                        "fact": "事实乙",
                        "reason": "理由乙",
                        "result": "",
                        "charge": ["罪名乙"],
                        "article": [2],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (source / "2.json").write_text(
                json.dumps(
                    {
                        "pid": 2,
                        "qw": "判决书甲",
                        "fact": "事实甲",
                        "reason": "理由甲",
                        "result": "判决甲",
                        "charge": ["罪名甲一", "罪名甲二"],
                        "article": [1],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output = temp_path / "corpus-full.jsonl"

            module = load_module()
            count = module.convert_corpus(source, output)

            raw_lines = output.read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in raw_lines]
            self.assertEqual(count, 2)
            self.assertEqual(
                records,
                [
                    {
                        "_id": "2",
                        "title": "",
                        "text": "判决书甲\n事实甲\n理由甲\n判决甲\n罪名甲一,罪名甲二\n",
                    },
                    {
                        "_id": "10",
                        "title": "",
                        "text": "判决书乙\n事实乙\n理由乙\n\n罪名乙\n",
                    },
                ],
            )
            self.assertIn("\\u5224\\u51b3\\u4e66", raw_lines[0])
            self.assertEqual(output.stat().st_mode & 0o666, 0o664)


if __name__ == "__main__":
    unittest.main()
