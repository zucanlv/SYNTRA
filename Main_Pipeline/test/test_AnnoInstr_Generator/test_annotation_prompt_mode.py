from pathlib import Path
import sys


MAIN_PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN_PIPELINE_DIR))

from AnnoInstr_Generator import (  # noqa: E402
    build_anno_instr_meta_prompt,
    validate_instruction,
)


def test_single_doc_no_doc_id_meta_prompt_uses_simple_output_contract():
    prompt = build_anno_instr_meta_prompt(
        query_type="web query",
        doc_type="web passage",
        relevance_def="answers the query",
        annotation_prompt_mode="single_doc_no_doc_id",
    )

    assert "Exactly one candidate document" in prompt
    assert "No doc_id or document identifier is provided" in prompt
    assert "positive_doc_ids" not in prompt
    assert '{"score": 0-3' in prompt


def test_meta_prompt_asks_for_task_specific_scoring_not_generic_web_answering():
    prompt = build_anno_instr_meta_prompt(
        query_type="entity-centric category query",
        doc_type="encyclopedic entity page",
        relevance_def="the entity belongs to the requested category",
        annotation_prompt_mode="multi_doc_with_doc_id",
    )

    assert "task-specific 0-3 absolute scoring rubric" in prompt
    assert "Do not use generic web-answering language" in prompt
    assert "valid-but-less-complete positive cases" in prompt
    assert "Partially or weakly satisfies" not in prompt
    assert "document directly and comprehensively answers the query" not in prompt
    assert "provides critical information" not in prompt


def test_single_doc_no_doc_id_validation_allows_grouped_fields_when_required_fields_exist():
    old_contract = (
        "Use a 0-3 scoring system. Output JSON with annotations, score, classification, reasoning, "
        "positive_doc_ids, hard_negative_doc_ids, easy_negative_doc_ids and doc_id."
    )

    ok, errors = validate_instruction(
        old_contract, annotation_prompt_mode="single_doc_no_doc_id"
    )

    assert ok, errors


def test_single_doc_no_doc_id_validation_rejects_truncated_instruction():
    truncated = (
        "Score 3 positive. Score 2 positive. Score 1 hard_negative: "
        "The document initially looks plausible and may superfic"
    )

    ok, errors = validate_instruction(
        truncated, annotation_prompt_mode="single_doc_no_doc_id"
    )

    assert not ok
    assert any("classification" in e for e in errors)
    assert any("reasoning" in e for e in errors)


def test_single_doc_no_doc_id_validation_accepts_simple_schema():
    new_contract = (
        "Evaluate one document using a 0-3 score. Output only valid JSON with "
        "score, classification, and reasoning. Do not output a doc_id."
    )

    ok, errors = validate_instruction(
        new_contract, annotation_prompt_mode="single_doc_no_doc_id"
    )

    assert ok, errors


if __name__ == "__main__":
    test_single_doc_no_doc_id_meta_prompt_uses_simple_output_contract()
    test_meta_prompt_asks_for_task_specific_scoring_not_generic_web_answering()
    test_single_doc_no_doc_id_validation_allows_grouped_fields_when_required_fields_exist()
    test_single_doc_no_doc_id_validation_rejects_truncated_instruction()
    test_single_doc_no_doc_id_validation_accepts_simple_schema()
