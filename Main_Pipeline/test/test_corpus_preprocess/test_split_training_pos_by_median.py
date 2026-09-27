import importlib.util
import json
import tempfile
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "corpus_preprocess"
    / "6-16-split-training-pos-by-median.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("split_training_pos_by_median", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_split_training_pos_by_median():
    module = load_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_jsonl = tmpdir / "training.jsonl"
        diverse_query = tmpdir / "DiverseQuery_Main.json"
        output_dir = tmpdir / "splits"

        entries = [
            {"query": "q1", "pos": ["p1", "origin1"], "neg": ["n1"]},
            {"query": "q2", "pos": ["a", "b"], "neg": ["n2"]},
            {"query": "q3", "pos": ["solo"], "neg": ["n3"]},
        ]
        with input_jsonl.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        diverse_query.write_text(
            json.dumps(
                {
                    "results": [
                        {"doc": "origin1", "queries": [{"query": "q1"}]},
                        {"doc": "missing-from-pos", "queries": [{"query": "q2"}]},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        query_to_origin_doc = module.load_query_to_origin_doc([diverse_query])
        pos_counts = module.collect_pos_counts(input_jsonl)
        split_count, median_value = module.choose_split_count(pos_counts, "floor")
        stats = module.write_splits(
            input_jsonl=input_jsonl,
            output_dir=output_dir,
            split_count=split_count,
            query_to_origin_doc=query_to_origin_doc,
        )

        assert median_value == 2
        assert split_count == 2
        assert module.choose_split_count([1, 2, 3, 4], "ceil") == (3, 2.5)
        assert stats["total"] == 3
        assert stats["origin_moved_to_first"] == 1
        assert stats["origin_not_in_pos"] == 1
        assert stats["query_not_in_mapping"] == 1
        assert stats["wrapped_entries"] == 1

        split1 = read_jsonl(output_dir / "split1.jsonl")
        split2 = read_jsonl(output_dir / "split2.jsonl")

        assert [entry["pos"] for entry in split1] == [["origin1"], ["a"], ["solo"]]
        assert [entry["pos"] for entry in split2] == [["p1"], ["b"], ["solo"]]
        assert [entry["neg"] for entry in split1] == [["n1"], ["n2"], ["n3"]]
        assert [entry["neg"] for entry in split2] == [["n1"], ["n2"], ["n3"]]


if __name__ == "__main__":
    test_split_training_pos_by_median()
    print("ok")
