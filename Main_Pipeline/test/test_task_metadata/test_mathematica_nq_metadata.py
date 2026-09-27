import hashlib
import importlib.util
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[2]

EXPECTED_DEFINITIONS = {
    "cqadupstack-mathematica": (
        "Mathematica Q&A question",
        "Mathematica Q&A forum post",
        "Given a query (Mathematica Q&A question) and a document (Mathematica "
        "Q&A forum post), the document is relevant if it asks or addresses the same "
        "underlying Mathematica or Wolfram Language question or problem as the query. "
        "It should express an equivalent or near-duplicate information need even if "
        "the wording, code, data, or concrete example differs. Merely mentioning the "
        "same function, symbol, or broad topic is insufficient unless the document "
        "helps resolve the query's specific concern."
    ),
    "nq": (
        "Natural-language factoid question",
        "Wikipedia passage",
        "Given a query (natural-language factoid question) and a document (Wikipedia "
        "passage), the document is relevant if it contains factual evidence that "
        "directly answers the question or provides the specific information needed "
        "to derive the answer. Merely discussing the same entity or broad topic "
        "without resolving the question is insufficient."
    ),
}

# SHA-256(query + NUL + Positive). IDs make the fixed seed-42 samples auditable
# against each dataset's official test qrels.
EXPECTED_FEW_SHOT_DIGESTS = {
    "cqadupstack-mathematica": [
        ("48681", "3964", "f83dbae3ecadf6b4144429487b0d4aea0bac51e670cd686262f1e8395441a998"),
        ("9772", "11155", "4a8981742afc91dba4d2436797ff0051909b62c0b18a820c6e5a835382940600"),
        ("57366", "57677", "f9466808c9c079f8565555bd22eca1ec89e4aac0d8ad1c58aafd06a970dc010f"),
        ("32658", "32378", "0ed6adce63fd6c31561fc2939330eaaf29e06b51180ea3dfe82951f1f1c92080"),
        ("16429", "7722", "d4a45d43f3dbbbfbfb33f51cf91089f51d890de84022ae7514bf5b28b0eef798"),
    ],
    "nq": [
        ("test2619", "doc89759", "f1c104dba10829edcc983e6e6403d25dfc6f2594fffef00be3504577da5486b7"),
        ("test456", "doc16797", "3e7a45ce3e4d7121c752edd4dfbc95db80fc7ec4a6b236c81d6fd4432c357792"),
        ("test102", "doc3294", "35facf0e85192782690dc1dd47600ea402b4b8163215ab4a98d739c40a55423a"),
        ("test3037", "doc104077", "05226b0ec12ded56dc4e3e8d4656565b7893a2803cdfb7c4d73a7c99d233defc"),
        ("test1126", "doc39696", "37e5d8733dfab8bfdc251bf52ce16acfd5c1f970b731acab196d4a6d135c1e10"),
    ],
}


def load_registry(filename, variable_name):
    path = PIPELINE_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, variable_name)


def test_mathematica_and_nq_high_level_definitions():
    definitions = load_registry("HighLevel_Def.py", "HighLevel_Task_Definition")

    for task, expected in EXPECTED_DEFINITIONS.items():
        assert definitions[task] == expected


def test_mathematica_and_nq_use_fixed_official_few_shots():
    examples = load_registry("Few_Shot_Example.py", "Few_Shot_Example")

    for task, expected_rows in EXPECTED_FEW_SHOT_DIGESTS.items():
        task_examples = examples[task]
        assert len(task_examples) == 5
        assert len({item["query"] for item in task_examples}) == 5
        actual_digests = [
            hashlib.sha256((item["query"] + "\0" + item["Positive"]).encode()).hexdigest()
            for item in task_examples
        ]
        assert actual_digests == [digest for *_, digest in expected_rows]
        for item in task_examples:
            assert set(item) == {"query", "Positive"}
            assert isinstance(item["query"], str) and item["query"].strip()
            assert isinstance(item["Positive"], str) and item["Positive"].strip()


if __name__ == "__main__":
    test_mathematica_and_nq_high_level_definitions()
    test_mathematica_and_nq_use_fixed_official_few_shots()
    print("PASS: 2 tests")
