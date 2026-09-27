from pathlib import Path
import sys


MAIN_PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN_PIPELINE_DIR))

from AnnoInstr_Generator import build_anno_instr_refine_prompt  # noqa: E402


def test_refine_prompt_can_omit_annotation_samples_for_calibration():
    prompt = build_anno_instr_refine_prompt(
        annotation_instruction="Current instruction",
        annotation_samples=[{"query": "q", "annotations": []}],
        feedback="Use stricter hard-negative guidance.",
        task_name="msmarco",
        annotation_prompt_mode="single_doc_no_doc_id",
        include_annotation_samples=False,
    )

    assert "Use stricter hard-negative guidance." in prompt
    assert "Current instruction" in prompt
    assert "Sample Annotations" not in prompt
    assert "single query-document annotation setting" in prompt
    assert "Output only one JSON object with score, classification, and reasoning" in prompt
    assert "positive_doc_ids" not in prompt


def test_refine_prompt_keeps_annotation_samples_by_default_for_non_calibration():
    prompt = build_anno_instr_refine_prompt(
        annotation_instruction="Current instruction",
        annotation_samples=[{"query": "q", "annotations": []}],
        feedback="Fix labels.",
        task_name="msmarco",
        annotation_prompt_mode="multi_doc_with_doc_id",
    )

    assert "Sample Annotations" in prompt
    assert "multi-document annotation setting" in prompt
    assert "positive_doc_ids" in prompt
    assert "exact doc_id echoing" in prompt


if __name__ == "__main__":
    test_refine_prompt_can_omit_annotation_samples_for_calibration()
    test_refine_prompt_keeps_annotation_samples_by_default_for_non_calibration()
