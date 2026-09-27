import importlib.util
import json
import tempfile
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "corpus_preprocess"
    / "evidencebench_100k_to_corpus.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("evidencebench_100k_to_corpus", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_dataset(path: Path, records: dict) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_write_test_corpora_preserves_pool_and_filters_generation_candidates(tmp_path):
    module = _load_module()
    source = tmp_path / "test.json"
    full_output = tmp_path / "sentences.jsonl"
    generation_output = tmp_path / "generation.jsonl"
    shared_sentence = (
        "The intervention significantly reduced symptoms in high-risk participants "
        "during the twelve month follow-up period."
    )
    records = {
        "test_0": {
            "hypothesis": "This test hypothesis must never enter the corpus schema.",
            "paper_id": "pmc_1",
            "paper_as_candidate_pool": [
                "Methods",
                shared_sentence,
                "Too short.",
            ],
            "sentence_types_in_candidate_pool": [
                "section name",
                "abstract",
                "normal paragraph",
            ],
            "sentence_index2aspects": {"0": [], "1": ["test_0_aspect_0"], "2": []},
        },
        "test_1": {
            "hypothesis": "A second hidden test hypothesis.",
            "paper_id": "pmc_2",
            "paper_as_candidate_pool": [
                shared_sentence,
                "Participants receiving standard care showed no measurable change over the complete study period.",
            ],
            "sentence_types_in_candidate_pool": ["normal paragraph", "normal paragraph"],
            "sentence_index2aspects": {"0": [], "1": []},
        },
    }
    _write_dataset(source, records)

    stats = module.write_test_corpora(
        source,
        full_output,
        generation_output,
        min_generation_words=10,
    )

    full_rows = _read_jsonl(full_output)
    assert [row["doc_id"] for row in full_rows] == [
        "pmc_1#000000",
        "pmc_1#000001",
        "pmc_1#000002",
        "pmc_2#000000",
        "pmc_2#000001",
    ]
    assert [row["text"] for row in full_rows] == [
        "Methods",
        shared_sentence,
        "Too short.",
        shared_sentence,
        "Participants receiving standard care showed no measurable change over the complete study period.",
    ]
    assert set(full_rows[0]) == {
        "doc_id",
        "title",
        "text",
        "paper_id",
        "sentence_idx",
        "sentence_type",
        "split",
    }
    assert all("hypothesis" not in row and "sentence_index2aspects" not in row for row in full_rows)

    generation_rows = _read_jsonl(generation_output)
    assert [row["doc_id"] for row in generation_rows] == [
        "pmc_1#000001",
        "pmc_2#000001",
    ]
    assert stats == {
        "records": 2,
        "unique_papers": 2,
        "duplicate_paper_records": 0,
        "sentences": 5,
        "generation_candidates": 2,
        "generation_duplicates_skipped": 1,
        "generation_ineligible_skipped": 2,
    }


def test_write_test_corpora_deduplicates_identical_repeated_papers(tmp_path):
    module = _load_module()
    source = tmp_path / "test.json"
    full_output = tmp_path / "sentences.jsonl"
    generation_output = tmp_path / "generation.jsonl"
    sentence = (
        "A repeated research paper reports the same sufficiently detailed biomedical "
        "sentence for two independently evaluated hypotheses."
    )
    base_record = {
        "paper_id": "pmc_repeated",
        "paper_as_candidate_pool": [sentence],
        "sentence_types_in_candidate_pool": ["normal paragraph"],
        "sentence_index2aspects": {"0": []},
    }
    records = {
        "test_0": {**base_record, "hypothesis": "The first hidden hypothesis."},
        "test_1": {**base_record, "hypothesis": "The second hidden hypothesis."},
    }
    _write_dataset(source, records)

    stats = module.write_test_corpora(
        source,
        full_output,
        generation_output,
        min_generation_words=10,
    )

    assert len(_read_jsonl(full_output)) == 1
    assert stats["records"] == 2
    assert stats["unique_papers"] == 1
    assert stats["duplicate_paper_records"] == 1


def test_select_few_shots_prefers_hypothesis_aligned_evidence_over_methods(tmp_path):
    module = _load_module()
    source = tmp_path / "train.json"
    hypothesis = (
        "Higher vitamin D supplementation reduces fracture risk among older adults."
    )
    outcome = (
        "Vitamin D supplementation significantly reduced fracture risk among older "
        "adults compared with placebo treatment."
    )
    methods = (
        "This multicenter retrospective cohort enrolled participants from twelve "
        "hospitals and followed them according to the approved protocol."
    )
    _write_dataset(
        source,
        {
            "train_0": {
                "hypothesis": hypothesis,
                "paper_id": "pmc_alignment",
                "systematic_review_id": "review_alignment",
                "paper_as_candidate_pool": [methods, outcome],
                "sentence_types_in_candidate_pool": [
                    "normal paragraph",
                    "normal paragraph",
                ],
                "sentence_index2aspects": {
                    "0": ["a0", "a1", "a2"],
                    "1": ["a3"],
                },
                "aspect_id2aspect": {
                    "a0": "The study was retrospective.",
                    "a1": "The study was multicenter.",
                    "a2": "The study enrolled participants from twelve hospitals.",
                    "a3": "Vitamin D reduced fracture risk compared with placebo.",
                },
                "aspect_list_ids": ["a0", "a1", "a2", "a3"],
                "results_aspect_list_ids": None,
                "results_evidence_retrieval_at_optimal_evaluation": None,
            }
        },
    )

    examples, _stats = module.select_few_shots(source, count=1)

    assert examples == [{"query": hypothesis, "Positive": outcome}]


def test_select_few_shots_uses_labeled_train_sentences_and_distinct_sources(tmp_path):
    module = _load_module()
    source = tmp_path / "train.json"
    records = {}
    for index in range(6):
        aspect_id = f"train_{index}_aspect_0"
        records[f"train_{index}"] = {
            "hypothesis": (
                f"Treatment number {index} improves recovery among adults with chronic respiratory disease."
            ),
            "paper_id": f"pmc_{index}",
            "systematic_review_id": f"review_{index}",
            "paper_as_candidate_pool": [
                "Background",
                (
                    f"In trial {index}, treated adults experienced significantly faster recovery "
                    "than participants who received standard care during follow-up."
                ),
            ],
            "sentence_types_in_candidate_pool": [
                "section name",
                "abstract" if index < 2 else "normal paragraph",
            ],
            "sentence_index2aspects": {"0": [], "1": [aspect_id]},
            "aspect_id2aspect": {aspect_id: f"Trial {index} reported faster recovery."},
            "aspect_list_ids": [aspect_id],
            "results_aspect_list_ids": None,
            "results_evidence_retrieval_at_optimal_evaluation": None,
        }
    _write_dataset(source, records)

    examples, stats = module.select_few_shots(source, count=5)

    assert len(examples) == 5
    assert all(set(example) == {"query", "Positive"} for example in examples)
    assert len({example["query"] for example in examples}) == 5
    assert all("significantly faster recovery" in example["Positive"] for example in examples)
    assert stats["records"] == 6
    assert stats["aspect_text_records"] == 6
    assert stats["nullable_results_aspect_records"] == 6
    assert stats["eligible_few_shot_records"] == 6

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        test_write_test_corpora_preserves_pool_and_filters_generation_candidates(
            Path(temp_dir)
        )
    with tempfile.TemporaryDirectory() as temp_dir:
        test_write_test_corpora_deduplicates_identical_repeated_papers(Path(temp_dir))
    with tempfile.TemporaryDirectory() as temp_dir:
        test_select_few_shots_prefers_hypothesis_aligned_evidence_over_methods(
            Path(temp_dir)
        )
    with tempfile.TemporaryDirectory() as temp_dir:
        test_select_few_shots_uses_labeled_train_sentences_and_distinct_sources(
            Path(temp_dir)
        )
    print("EvidenceBench conversion tests passed")

