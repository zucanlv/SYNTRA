import importlib.util
import pathlib
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "corpus_preprocess"
    / "split_queries_by_position.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("split_queries_by_position", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SplitQueriesByPositionTest(unittest.TestCase):
    def test_keeps_only_the_requested_query_for_each_document(self):
        module = load_module()
        payload = {
            "input_path": "source.json",
            "num_queries": 2,
            "use_idattr": True,
            "results": [
                {
                    "doc_id": "doc-a",
                    "doc": "first document",
                    "queries": [{"query": "a-first"}, {"query": "a-second"}],
                },
                {
                    "doc_id": "doc-b",
                    "doc": "second document",
                    "queries": [{"query": "b-first"}, {"query": "b-second"}],
                },
            ],
        }

        split = module.split_payload_by_query_position(payload, query_position=2)

        self.assertEqual(split["num_queries"], 1)
        self.assertEqual(
            [item["queries"] for item in split["results"]],
            [[{"query": "a-second"}], [{"query": "b-second"}]],
        )
        self.assertEqual(
            [item["doc_id"] for item in split["results"]], ["doc-a", "doc-b"]
        )
        self.assertEqual(payload["results"][0]["queries"], [{"query": "a-first"}, {"query": "a-second"}])


if __name__ == "__main__":
    unittest.main()
