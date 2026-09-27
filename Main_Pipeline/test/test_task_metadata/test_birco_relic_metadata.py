import hashlib
import importlib.util
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[2]

EXPECTED_DEFINITION = (
    "Literary analysis excerpt with a masked quotation",
    "Candidate passage from a literary work",
    "Given a query (an excerpt of literary analysis containing a quotation "
    "replaced by '[masked sentence(s)]') and a document (a candidate passage "
    "from the analyzed literary work), the document is relevant if it can be "
    "naturally inserted at the masked position and directly supports at least "
    "one of the literary claims made in the surrounding context. Sharing the "
    "same work, characters, themes, or vocabulary without supporting the "
    "surrounding interpretation, or merely repeating content already present "
    "in the query, is insufficient.",
)

# SHA-256(query + NUL + Positive). Source IDs make the fixed seed-42 samples
# auditable against mteb/BIRCO-Relic-Test.
EXPECTED_FEW_SHOT_DIGESTS = [
    ("q_10200", "c_1706213", "63385ac62d2185e600b471aa760b33e8675032871fe2a4e5bd6b78fdb69b3c82"),
    ("q_3721", "c_588391", "9825b839c40b5afb65e398164bedfcd154adb05a18fee398443bd4a378cf13e6"),
    ("q_14148", "c_2817930", "53ca4323696c37c1cd4eb25740edf16db90fe6f3a7d5bd3dd3b5aa18eaf5df64"),
    ("q_15326", "c_3131160", "cbf4e289227b26512196dee7530012525c9628d95af5e2fcc95acc8b1c72dd9a"),
    ("q_9313", "c_1480007", "f67b7e0c0f242fe297d372d18d1525dc348e4b4a9aa5dc6a51234f0ffe80241a"),
]


def load_registry(filename, variable_name):
    path = PIPELINE_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, variable_name)


def test_birco_relic_high_level_definition():
    definitions = load_registry("HighLevel_Def.py", "HighLevel_Task_Definition")

    assert definitions["birco-relic"] == EXPECTED_DEFINITION


def test_birco_relic_uses_fixed_official_few_shots():
    examples = load_registry("Few_Shot_Example.py", "Few_Shot_Example")
    task_examples = examples["birco-relic"]

    assert len(task_examples) == 5
    assert len({item["query"] for item in task_examples}) == 5
    actual_digests = [
        hashlib.sha256((item["query"] + "\0" + item["Positive"]).encode()).hexdigest()
        for item in task_examples
    ]
    assert actual_digests == [digest for *_, digest in EXPECTED_FEW_SHOT_DIGESTS]
    for item in task_examples:
        assert set(item) == {"query", "Positive"}
        assert "[masked sentence(s)]" in item["query"]
        assert all(isinstance(value, str) and value.strip() for value in item.values())


if __name__ == "__main__":
    test_birco_relic_high_level_definition()
    test_birco_relic_uses_fixed_official_few_shots()
    print("PASS: 2 tests")
