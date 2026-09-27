import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from annotation_app.server import build_server


class ServerVendorTest(unittest.TestCase):
    def test_vendor_font_subdirectory_is_served_without_exposing_other_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "web" / "vendor" / "fonts").mkdir(parents=True)
            (root / "data" / "samples.json").write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "theoremqa-theorems-001",
                            "dataset": "theoremqa-theorems",
                            "query": "q",
                            "document": "d",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (root / "data" / "translations.zh-CN.json").write_text(
                json.dumps(
                    {
                        "theoremqa-theorems-001": {"query": "查询", "document": "文档"}
                    }
                ),
                encoding="utf-8",
            )
            font = root / "web" / "vendor" / "fonts" / "KaTeX_Main-Regular.woff2"
            font.write_bytes(b"font-data")
            server = build_server(root, "annotator_1", host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                with urllib.request.urlopen(
                    f"http://{host}:{port}/vendor/fonts/KaTeX_Main-Regular.woff2"
                ) as response:
                    self.assertEqual(response.read(), b"font-data")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
