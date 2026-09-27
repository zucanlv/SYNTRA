import importlib.util
import json
import tempfile
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "corpus_preprocess"
    / "legalbenchrag_to_corpus.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("legalbenchrag_to_corpus", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_corpus_writes_sorted_relative_ids_and_document_text(tmp_path):
    source = tmp_path / "LegalBench-RAG"
    (source / "cuad").mkdir(parents=True)
    (source / "contractnli").mkdir()
    (source / "cuad" / "zeta.txt").write_text(
        "Clause one.\nClause two.\n", encoding="utf-8"
    )
    (source / "contractnli" / "alpha.txt").write_text(
        "  Mutual NDA.  \n", encoding="utf-8"
    )
    (source / "README.md").write_text("not corpus content", encoding="utf-8")
    output = tmp_path / "legalbenchrag_corpus.jsonl"

    module = load_module()
    count = module.convert_corpus(source, output)

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert count == 2
    assert records == [
        {"id": "contractnli/alpha.txt", "text": "Mutual NDA."},
        {"id": "cuad/zeta.txt", "text": "Clause one.\nClause two."},
    ]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        test_convert_corpus_writes_sorted_relative_ids_and_document_text(
            Path(temp_dir)
        )
    print("PASS")
