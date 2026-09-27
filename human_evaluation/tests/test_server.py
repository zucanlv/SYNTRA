import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from annotation_app.server import build_server


def make_samples():
    samples = []
    for dataset in ("msmarco", "text2sql", "theoremqa-theorems"):
        for index in range(6):
            samples.append(
                {
                    "sample_id": f"{dataset}-{index:03d}",
                    "dataset": dataset,
                    "query": f"query {index}",
                    "document": f"document {index}",
                }
            )
    return samples


class RunningServer:
    def __init__(self, root: Path, annotator: str):
        self.server = build_server(root, annotator, host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __exit__(self, exc_type, exc, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "data").mkdir()
        (self.root / "web").mkdir()
        (self.root / "guidelines").mkdir()
        (self.root / "data" / "samples.json").write_text(
            json.dumps(make_samples()), encoding="utf-8"
        )
        translations = {
            sample["sample_id"]: {
                "query": f"查询 {sample['sample_id']}",
                "document": f"文档 {sample['sample_id']}",
            }
            for sample in make_samples()
        }
        (self.root / "data" / "translations.zh-CN.json").write_text(
            json.dumps(translations, ensure_ascii=False), encoding="utf-8"
        )
        (self.root / "web" / "index.html").write_text("<h1>Annotation</h1>", encoding="utf-8")
        for dataset in ("msmarco", "text2sql", "theoremqa-theorems"):
            (self.root / "guidelines" / f"{dataset}.md").write_text(
                f"# {dataset}", encoding="utf-8"
            )

    def tearDown(self):
        self.temporary.cleanup()

    def test_session_groups_datasets_and_order_is_annotator_specific(self):
        orders = []
        for annotator in ("annotator_1", "annotator_2"):
            with RunningServer(self.root, annotator) as base_url:
                with urllib.request.urlopen(f"{base_url}/api/session") as response:
                    payload = json.load(response)
            self.assertEqual(
                payload["dataset_order"],
                ["msmarco", "text2sql", "theoremqa-theorems"],
            )
            self.assertEqual(payload["scores"], {})
            self.assertIn("query_zh", payload["samples"][0])
            self.assertIn("document_zh", payload["samples"][0])
            orders.append([sample["sample_id"] for sample in payload["samples"]])
        self.assertNotEqual(orders[0][:6], orders[1][:6])
        self.assertEqual(orders[0][:6], [item for item in orders[0] if item.startswith("msmarco")])

    def test_post_annotation_writes_csv_before_success_response(self):
        with RunningServer(self.root, "annotator_1") as base_url:
            request = urllib.request.Request(
                f"{base_url}/api/annotations",
                data=json.dumps({"sample_id": "msmarco-000", "score": 3}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                payload = json.load(response)
            self.assertEqual(payload["saved_score"], 3)
            self.assertEqual(payload["completed"], 1)
            csv_text = (self.root / "annotations" / "annotator_1.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("msmarco-000,msmarco,3", csv_text)

    def test_static_path_traversal_is_rejected(self):
        with RunningServer(self.root, "annotator_1") as base_url:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(f"{base_url}/..%2Fdata%2Fsamples.json")
            self.assertEqual(caught.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
