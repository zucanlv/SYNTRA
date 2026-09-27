import hashlib
import importlib.util
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[2]

EXPECTED_DEFINITION = (
    "Natural-language question about a legal contract or privacy policy",
    "Legal contract or privacy policy passage",
    "Given a query (a natural-language question about a legal contract or "
    "privacy policy) and a document (a legal contract or privacy policy "
    "passage), the document is relevant if it contains one or more specific "
    "clauses or textual statements that directly answer the question or provide "
    "the evidence needed to determine the answer. Evidence that establishes a "
    "negative answer is also relevant; merely discussing the same legal topic, "
    "agreement, party, or policy without resolving the question is insufficient.",
)

# Fixed examples from the official LegalBench-RAG benchmark JSON files. Each row
# records (subset, benchmark index, corpus file, SHA-256(query + NUL + Positive)).
EXPECTED_FEW_SHOT_ROWS = [
    (
        "contractnli",
        509,
        "contractnli/oceaneering-non-disclosure-agreement.txt",
        "4d1364f10ed7a7b3e3780ff01069f733ec66727982d2380295d8f3a685ea3bdf",
    ),
    (
        "cuad",
        2450,
        "cuad/IMMUNOMEDICSINC_08_07_2019-EX-10.1-PROMOTION AGREEMENT.txt",
        "8f10c21535ce2f3838b54439dd5343c102aebe6ec37dc874206599cde1cff4d3",
    ),
    (
        "maud",
        1446,
        "maud/WPX Energy, Inc._Devon Energy Corporation.txt",
        "38bb61b57a753165c41f38c0448ce56ad09b236368232b01ade31667871f5be9",
    ),
    (
        "privacy_qa",
        148,
        "privacy_qa/23andMe.txt",
        "5d64d0729d96ce596c46bc8b2b39041dcc7d0aa98d50d77a54fb032a5d213cbd",
    ),
    (
        "contractnli",
        209,
        "contractnli/JB-Machine-LLC-NDA-1.txt",
        "e3cdc3829693b6088271c0bfa74f977446536cee1895673bb40637232918c44b",
    ),
    (
        "cuad",
        1721,
        "cuad/ADUROBIOTECH,INC_06_02_2020-EX-10.7-CONSULTING AGREEMENT(1).txt",
        "fda73a2fdf69f556783db6003568fef9b736130973b43a37749ee7db69d903ab",
    ),
    (
        "maud",
        918,
        "maud/Contango_Oil_&_Gas_KKR_&_Co.txt",
        "3a93d31706c4a1377ba4511cc574bc353d549748a62281285c0a6aaf8e585963",
    ),
    (
        "privacy_qa",
        107,
        "privacy_qa/Wordscapes.txt",
        "3a2f060b91199d7c056ffa73354f7ee3fed0f945af1c825873a45466f82a54d6",
    ),
]


def load_registry(filename, variable_name):
    path = PIPELINE_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, variable_name)


def test_legalbench_rag_high_level_definition():
    definitions = load_registry("HighLevel_Def.py", "HighLevel_Task_Definition")

    assert definitions["legalbench-rag"] == EXPECTED_DEFINITION


def test_legalbench_rag_uses_fixed_official_examples_from_every_subset():
    examples = load_registry("Few_Shot_Example.py", "Few_Shot_Example")
    task_examples = examples["legalbench-rag"]

    assert len(task_examples) == len(EXPECTED_FEW_SHOT_ROWS)
    assert len({item["query"] for item in task_examples}) == len(task_examples)
    assert {row[0] for row in EXPECTED_FEW_SHOT_ROWS[:4]} == {
        "contractnli",
        "cuad",
        "maud",
        "privacy_qa",
    }
    actual_digests = [
        hashlib.sha256(
            (item["query"] + "\0" + item["Positive"]).encode()
        ).hexdigest()
        for item in task_examples
    ]
    assert actual_digests == [row[3] for row in EXPECTED_FEW_SHOT_ROWS]
    for item in task_examples:
        assert set(item) == {"query", "Positive"}
        assert isinstance(item["query"], str) and item["query"].strip()
        assert isinstance(item["Positive"], str) and item["Positive"].strip()


if __name__ == "__main__":
    test_legalbench_rag_high_level_definition()
    test_legalbench_rag_uses_fixed_official_examples_from_every_subset()
    print("PASS: 2 tests")
