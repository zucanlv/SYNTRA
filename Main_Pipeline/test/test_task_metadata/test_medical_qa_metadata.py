import hashlib
import importlib.util
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[2]

EXPECTED_DEFINITION = (
    "English-language medical question",
    "English-language medical answer passage",
    "Given a query (English-language medical question) and a document "
    "(English-language medical answer passage), the document is relevant if "
    "it directly and accurately addresses the specific medical information "
    "need about the named disease or condition, such as its definition, "
    "symptoms, diagnosis, treatment, risk factors, prevention, or prognosis. "
    "Merely mentioning the same condition without resolving the aspect asked "
    "about in the query is insufficient."
)

# SHA-256(query + NUL + Positive) for curated pairs from the official test qrels.
# IDs make the exact source rows auditable against mteb/medical_qa.
EXPECTED_FEW_SHOT_DIGESTS = [
    (
        "d02b0cf8-a0f0-4508-acdc-37a6ff6a3f0c",
        "bed2a585-886f-43ed-9ecd-6c60af9b9696",
        "dac866e041240eb5ba29244fd6d1935ac250fb9aad8fbe0d050e480746acb8b3",
    ),
    (
        "0f6fcc6a-a221-49fc-bd65-c51459b722b9",
        "bb14fc4e-c6d3-4d64-add4-26ea26bb70fe",
        "9423fbaf0fa4d37ca903fe20a7baf203c1ec1045a703c7096280c5b07ad5457b",
    ),
    (
        "7cd94a58-b4f4-4c04-a705-b2b6a9043a72",
        "8f171388-acf3-4906-b379-8b1746456f2e",
        "2fc76c75f53a6abd097e2a7cafd23ab548ef00a32d442c8e356809fa86587d4c",
    ),
    (
        "25c8206a-1733-4616-8a86-4da10cb42601",
        "b26613b9-8006-47a3-bfc5-8ef00b79e580",
        "821f87b98eb921e52b1cda9889869425045a0ce9afca5473dc2b6bb91af43976",
    ),
    (
        "342be538-ac65-44cf-aa50-da4d1cb6e3f6",
        "246f6a07-3510-47bf-8dcc-a88e301bc0e4",
        "79f81f7881c7fe6b9aabf0f6c254941ee4b847444cf77012449d6d2460d742b0",
    ),
]


def load_registry(filename, variable_name):
    path = PIPELINE_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, variable_name)


def test_medical_qa_high_level_definition():
    definitions = load_registry("HighLevel_Def.py", "HighLevel_Task_Definition")

    assert definitions["medical_qa"] == EXPECTED_DEFINITION


def test_medical_qa_uses_fixed_official_few_shots():
    examples = load_registry("Few_Shot_Example.py", "Few_Shot_Example")
    task_examples = examples["medical_qa"]

    assert len(task_examples) == 5
    assert len({item["query"] for item in task_examples}) == 5
    actual_digests = [
        hashlib.sha256((item["query"] + "\0" + item["Positive"]).encode()).hexdigest()
        for item in task_examples
    ]
    assert actual_digests == [digest for _, _, digest in EXPECTED_FEW_SHOT_DIGESTS]
    for item in task_examples:
        assert set(item) == {"query", "Positive"}
        assert isinstance(item["query"], str) and item["query"].strip()
        assert isinstance(item["Positive"], str) and item["Positive"].strip()


if __name__ == "__main__":
    test_medical_qa_high_level_definition()
    test_medical_qa_uses_fixed_official_few_shots()
    print("PASS: 2 tests")
