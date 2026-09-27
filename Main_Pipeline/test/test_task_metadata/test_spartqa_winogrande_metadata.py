import importlib.util
import hashlib
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[2]

EXPECTED_DEFINITIONS = {
    "spartqa-mchoice": (
        "Textual spatial scene followed by a multiple-choice question",
        "Candidate spatial answer phrase",
        "Given a query (a textual spatial scene followed by a multiple-choice "
        "question) and a document (a candidate spatial answer phrase), the document "
        "is relevant if it denotes an answer supported by the scene's stated spatial "
        "relations and valid inferences across objects and containers. If both "
        "choices satisfy the queried relation, each choice and 'both of them' are "
        "relevant; if neither does, 'none of them' is relevant. Mere object mention "
        "or lexical overlap is insufficient.",
    ),
    "winogrande": (
        "Commonsense fill-in-the-blank sentence",
        "Candidate answer phrase",
        "Given a query (a sentence containing one blank marked by an underscore) "
        "and a document (a candidate answer phrase), the document is relevant if "
        "inserting it into the blank yields the intended completion under the causal, "
        "physical, social, or temporal commonsense constraints of the full sentence. "
        "Grammatical compatibility or lexical association alone is insufficient.",
    ),
}

# SHA-256(query + NUL + Positive) for fixed seed-42 samples from MTEB test qrels.
# IDs are included to make the exact source rows auditable.
EXPECTED_FEW_SHOT_DIGESTS = {
    "spartqa-mchoice": [
        ("spartqa-q-3355", "spartqa-d-3", "f3df240ce337f59c8ca6ef7e08a93f727e75a74f068b9a3e8981efd952b37f7b"),
        ("spartqa-q-1408", "spartqa-d-0", "8fa606de3fd9b7e015a5b397498948da6ef8f8a59f4274c7ae1719a4b53254d9"),
        ("spartqa-q-109", "spartqa-d-3", "787278b2426b456dd5015103c099dea7d302b0ede2deb81c26256cdb77afc748"),
        ("spartqa-q-498", "spartqa-d-30", "fe53f301b984edcfe503205a1ab955ef3640d145304cba6b9e18a09017513c81"),
        ("spartqa-q-2010", "spartqa-d-35", "8c097722ce9da11eb156d6c70cac3cbdf9e6c4ccfeb993c26433e40dfd553eae"),
    ],
    "winogrande": [
        ("wino-q-1202", "wino-d-72", "5c3b32d0eeb910e89675849d5f23a46fe013c31b1c595d42a3ed5a5c33a50bb2"),
        ("wino-q-1043", "wino-d-77", "09eb2ad0194edbbc2e3a0adb058b3bbbf2ad433560bc949ad7491a3ee8ec465c"),
        ("wino-q-365", "wino-d-261", "8395bc621314c7de008976aeb855f3051e56d80038869dd0a390642480bf6e8c"),
        ("wino-q-309", "wino-d-7", "57c645b496b2b6b0b7b51c460224ed6222911cf4ba09a13b6e1f3c4c12b0939c"),
        ("wino-q-27", "wino-d-35", "2c560e3bf35de20db52729c358a230d05cf6df72e3af2afe9e13fe6f8164a2a9"),
    ],
}


def load_registry(filename, variable_name):
    path = PIPELINE_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, variable_name)


def test_spartqa_and_winogrande_high_level_definitions():
    definitions = load_registry("HighLevel_Def.py", "HighLevel_Task_Definition")

    for task, expected in EXPECTED_DEFINITIONS.items():
        assert definitions[task] == expected


def test_spartqa_and_winogrande_use_fixed_mteb_test_few_shots():
    examples = load_registry("Few_Shot_Example.py", "Few_Shot_Example")

    for task, expected_rows in EXPECTED_FEW_SHOT_DIGESTS.items():
        task_examples = examples[task]
        assert len(task_examples) == 5
        assert len({item["query"] for item in task_examples}) == 5
        actual_digests = [
            hashlib.sha256((item["query"] + "\0" + item["Positive"]).encode()).hexdigest()
            for item in task_examples
        ]
        assert actual_digests == [digest for _, _, digest in expected_rows]
        for item in task_examples:
            assert set(item) == {"query", "Positive"}
            assert isinstance(item["query"], str) and item["query"].strip()
            assert isinstance(item["Positive"], str) and item["Positive"].strip()


if __name__ == "__main__":
    test_spartqa_and_winogrande_high_level_definitions()
    test_spartqa_and_winogrande_use_fixed_mteb_test_few_shots()
    print("PASS: 2 tests")
