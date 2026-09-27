from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_compare_annotations_against_calibration_reports_mismatches():
    from AnnoCalibration_Generator import compare_annotations_against_calibration

    calibration_samples = [
        {
            "query_id": "q1",
            "query": "sample query",
            "candidates": [
                {"doc_id": "p1", "doc": "positive", "expected_label": "positive"},
                {"doc_id": "h1", "doc": "hard", "expected_label": "hard_negative"},
                {"doc_id": "e1", "doc": "easy", "expected_label": "easy_negative"},
            ],
        }
    ]
    annotation_samples = [
        {
            "query_id": "q1",
            "query": "sample query",
            "annotations": [
                {"doc_id": "p1", "classification": "positive"},
                {"doc_id": "h1", "classification": "positive"},
                {"doc_id": "e1", "classification": "easy negative"},
            ],
        }
    ]

    passed, feedback, details = compare_annotations_against_calibration(
        annotation_samples, calibration_samples
    )

    assert not passed
    assert details["total"] == 3
    assert details["correct"] == 2
    assert details["mismatches"][0]["doc_id"] == "h1"
    assert "expected hard_negative, got positive" in feedback


def test_compare_annotations_against_calibration_posneg_accepts_negative_subtype_swaps():
    from AnnoCalibration_Generator import compare_annotations_against_calibration

    calibration_samples = [
        {
            "query_id": "q1",
            "query": "sample query",
            "candidates": [
                {"doc_id": "p1", "doc": "positive", "expected_label": "positive"},
                {"doc_id": "h1", "doc": "hard", "expected_label": "hard_negative"},
                {"doc_id": "e1", "doc": "easy", "expected_label": "easy_negative"},
            ],
        }
    ]
    annotation_samples = [
        {
            "query_id": "q1",
            "query": "sample query",
            "annotations": [
                {"doc_id": "p1", "classification": "positive"},
                {"doc_id": "h1", "classification": "easy_negative"},
                {"doc_id": "e1", "classification": "hard_negative"},
            ],
        }
    ]

    strict_passed, _, strict_details = compare_annotations_against_calibration(
        annotation_samples, calibration_samples
    )
    posneg_passed, feedback, posneg_details = compare_annotations_against_calibration(
        annotation_samples, calibration_samples, label_granularity="posneg"
    )

    assert not strict_passed
    assert strict_details["correct"] == 1
    assert posneg_passed
    assert feedback == "All 3 calibration labels matched expected labels."
    assert posneg_details["correct"] == 3
    assert posneg_details["label_granularity"] == "posneg"


def test_compare_annotations_against_calibration_posneg_still_rejects_pos_neg_boundary_errors():
    from AnnoCalibration_Generator import compare_annotations_against_calibration

    calibration_samples = [
        {
            "query_id": "q1",
            "query": "sample query",
            "candidates": [
                {"doc_id": "p1", "doc": "positive", "expected_label": "positive"},
                {"doc_id": "h1", "doc": "hard", "expected_label": "hard_negative"},
            ],
        }
    ]
    annotation_samples = [
        {
            "query_id": "q1",
            "query": "sample query",
            "annotations": [
                {"doc_id": "p1", "classification": "hard_negative"},
                {"doc_id": "h1", "classification": "positive"},
            ],
        }
    ]

    passed, feedback, details = compare_annotations_against_calibration(
        annotation_samples, calibration_samples, label_granularity="posneg"
    )

    assert not passed
    assert details["total"] == 2
    assert details["correct"] == 0
    assert details["mismatches"][0]["expected"] == "positive"
    assert details["mismatches"][0]["predicted"] == "negative"
    assert details["mismatches"][0]["predicted_original"] == "hard_negative"
    assert "expected positive, got negative" in feedback


def test_build_calibration_refinement_feedback_prompt_limits_llm_to_feedback():
    from AnnoCalibration_Generator import build_calibration_refinement_feedback_prompt

    details = {
        "total": 1,
        "correct": 0,
        "mismatches": [
            {
                "query_id": "q1",
                "query": "sample query",
                "doc_id": "h1",
                "expected": "hard_negative",
                "predicted": "positive",
                "doc": "near miss",
                "expected_reasoning": "mentions the topic but does not answer the query",
                "predicted_reasoning": "same topic",
            }
        ],
    }

    prompt = build_calibration_refinement_feedback_prompt(
        annotation_instruction="Judge relevance.",
        task_name="msmarco",
        compare_details=details,
    )

    assert "Do not re-grade the calibration set" in prompt
    assert "Analyze each mismatch" in prompt
    assert "structured feedback that guides the instruction refiner's thinking" in prompt
    assert "Do not merely enumerate individual errors" in prompt
    assert "global failure pattern" in prompt
    assert '"feedback"' in prompt
    assert "positive_issues" in prompt
    assert "negative_issues" in prompt
    assert "instruction_gaps" in prompt
    assert "suggestions" in prompt
    assert "near miss" in prompt
    assert "same topic" in prompt
    assert "Annotation Samples" not in prompt
    assert "Confirmed Calibration Samples" not in prompt


def test_build_draft_samples_from_retrieved_results_uses_positive_and_candidate_pools():
    from AnnoCalibration_Generator import build_draft_samples_from_retrieved_results

    retrieved_results = [
        {
            "doc_id": "gold_1_pos",
            "doc": "known positive text",
            "queries": [
                {
                    "query": "gold query",
                    "candidates": [
                        {"doc_id": "gold_1_pos", "doc": "known positive text", "rank": 1},
                        {"doc_id": "d1", "doc": "near miss 1", "rank": 2},
                        {"doc_id": "d2", "doc": "near miss 2", "rank": 3},
                        {"doc_id": "d3", "doc": "far miss 1", "rank": 49},
                        {"doc_id": "d4", "doc": "far miss 2", "rank": 50},
                    ],
                }
            ],
        }
    ]

    labels = {
        "gold_1_pos": {
            "d1": {"classification": "hard_negative", "score": 1, "reasoning": "near miss"},
            "d2": {"classification": "hard_negative", "score": 1, "reasoning": "near miss"},
            "d3": {"classification": "easy_negative", "score": 0, "reasoning": "off topic"},
            "d4": {"classification": "easy_negative", "score": 0, "reasoning": "off topic"},
        }
    }
    draft = build_draft_samples_from_retrieved_results(
        task="msmarco",
        retrieved_results=retrieved_results,
        candidate_labels_by_query=labels,
        hard_pool_size=2,
        easy_pool_size=1,
    )

    assert draft["status"] == "draft"
    sample = draft["samples"][0]
    assert sample["positive"]["expected_label"] == "positive"
    assert [c["doc_id"] for c in sample["hard_negative_candidates"]] == ["d1", "d2"]
    assert [c["doc_id"] for c in sample["easy_negative_candidates"]] == ["d3"]
    assert sample["hard_negative_candidates"][0]["annotator_reasoning"] == "near miss"



def test_calibration_labeler_prompt_does_not_use_annoinstr():
    from AnnoCalibration_Generator import _build_calibration_labeler_system_prompt

    prompt = _build_calibration_labeler_system_prompt("msmarco")

    assert "This is NOT using an existing annotation instruction" in prompt
    assert "Relevance Definition" in prompt
    assert "exactly one candidate document" in prompt
    assert "Do not output a doc_id" in prompt
    assert "AnnoInstr" not in prompt



def test_calibration_labeler_user_prompt_contains_one_candidate_only():
    from AnnoCalibration_Generator import _build_calibration_labeler_user_prompt

    prompt = _build_calibration_labeler_user_prompt(
        "gold query", {"doc_id": "d1", "doc": "candidate text"}
    )

    assert "Candidate document:" in prompt
    assert "Candidate documents:" not in prompt
    assert "[doc_id: d1]" not in prompt
    assert "candidate text" in prompt



def test_annotate_candidates_for_calibration_uses_concurrent_runner():
    import types
    import sys
    import AnnoCalibration_Generator as mod

    calls = {}

    def fake_run_concurrent(fn, kwargs_list, enabled=True, max_workers=None, jitter=0.0, **_):
        calls["enabled"] = enabled
        calls["max_workers"] = max_workers
        calls["jitter"] = jitter
        calls["count"] = len(kwargs_list)
        return [fn(**kwargs) for kwargs in kwargs_list]

    def fake_annotate_one_candidate_for_calibration(task, query, candidate, *, llm_cfg, max_retries=1):
        return {
            "classification": "hard_negative",
            "score": 1,
            "reasoning": f"labeled {candidate['doc_id']}",
        }

    old_concurrent_runner = sys.modules.get("concurrent_runner")
    old_annotate_one = mod.annotate_one_candidate_for_calibration
    sys.modules["concurrent_runner"] = types.SimpleNamespace(run_concurrent=fake_run_concurrent)
    mod.annotate_one_candidate_for_calibration = fake_annotate_one_candidate_for_calibration
    try:
        labels = mod.annotate_candidates_for_calibration(
            "msmarco",
            "query",
            [{"doc_id": "d1", "doc": "doc 1"}, {"doc_id": "d2", "doc": "doc 2"}],
            llm_cfg={"model": "fake"},
            concurrency_enabled=True,
            max_workers=3,
            jitter=0.2,
        )
    finally:
        mod.annotate_one_candidate_for_calibration = old_annotate_one
        if old_concurrent_runner is None:
            sys.modules.pop("concurrent_runner", None)
        else:
            sys.modules["concurrent_runner"] = old_concurrent_runner

    assert calls == {"enabled": True, "max_workers": 3, "jitter": 0.2, "count": 2}
    assert labels["d1"]["reasoning"] == "labeled d1"
    assert labels["d2"]["classification"] == "hard_negative"


def test_confirm_draft_uses_human_selected_hard_and_easy_negatives():
    from AnnoCalibration_Generator import confirm_draft_from_human_selection

    draft = {
        "task": "msmarco",
        "status": "draft",
        "samples": [
            {
                "query_id": "q1",
                "query": "gold query",
                "positive": {"doc_id": "p1", "doc": "pos", "expected_label": "positive"},
                "hard_negative_candidates": [
                    {"doc_id": "h1", "doc": "hard 1"},
                    {"doc_id": "h2", "doc": "hard 2"},
                ],
                "easy_negative_candidates": [
                    {"doc_id": "e1", "doc": "easy 1"},
                ],
                "human_review": {
                    "selected_hard_negative_doc_id": "h2",
                    "selected_easy_negative_doc_id": "e1",
                },
            }
        ],
    }

    confirmed = confirm_draft_from_human_selection(draft)

    assert confirmed["status"] == "confirmed"
    candidates = confirmed["samples"][0]["candidates"]
    assert [(c["doc_id"], c["expected_label"]) for c in candidates] == [
        ("p1", "positive"),
        ("h2", "hard_negative"),
        ("e1", "easy_negative"),
    ]


def test_build_positive_only_calibration_from_fewshot_uses_query_positive_pairs():
    import AnnoCalibration_Generator as mod

    previous = mod.Few_Shot_Example.get("unit_task")
    mod.Few_Shot_Example["unit_task"] = [
        {"query": "query one", "Positive": "positive one", "Hard Negative": "ignore me"},
        {"query": "query two", "Positive": "positive two"},
        {"query": "missing positive", "Positive": ""},
    ]
    try:
        calibration = mod.build_positive_only_calibration_from_fewshot(
            "unit_task",
            golden_query_count=2,
        )
    finally:
        if previous is None:
            mod.Few_Shot_Example.pop("unit_task", None)
        else:
            mod.Few_Shot_Example["unit_task"] = previous

    assert calibration["status"] == "confirmed"
    assert calibration["source"] == "few_shot_positive_only"
    assert len(calibration["samples"]) == 2
    first = calibration["samples"][0]
    assert first["query"] == "query one"
    assert len(first["candidates"]) == 1
    assert first["candidates"][0]["doc"] == "positive one"
    assert first["candidates"][0]["expected_label"] == "positive"
    assert first["candidates"][0]["source"] == "few_shot_positive_only"


def test_positive_only_calibration_compare_requires_positive_label():
    import AnnoCalibration_Generator as mod

    previous = mod.Few_Shot_Example.get("unit_task_positive_compare")
    mod.Few_Shot_Example["unit_task_positive_compare"] = [
        {"query": "query one", "Positive": "positive one"},
    ]
    try:
        calibration = mod.build_positive_only_calibration_from_fewshot("unit_task_positive_compare")
    finally:
        if previous is None:
            mod.Few_Shot_Example.pop("unit_task_positive_compare", None)
        else:
            mod.Few_Shot_Example["unit_task_positive_compare"] = previous

    sample = calibration["samples"][0]
    doc_id = sample["candidates"][0]["doc_id"]

    passed, feedback, details = mod.compare_annotations_against_calibration(
        [
            {
                "query_id": sample["query_id"],
                "query": sample["query"],
                "annotations": [{"doc_id": doc_id, "classification": "hard_negative"}],
            }
        ],
        calibration["samples"],
    )

    assert not passed
    assert details["total"] == 1
    assert details["correct"] == 0
    assert details["mismatches"][0]["expected"] == "positive"
    assert "expected positive, got hard_negative" in feedback


if __name__ == "__main__":
    test_compare_annotations_against_calibration_reports_mismatches()
    test_compare_annotations_against_calibration_posneg_accepts_negative_subtype_swaps()
    test_compare_annotations_against_calibration_posneg_still_rejects_pos_neg_boundary_errors()
    test_build_calibration_refinement_feedback_prompt_limits_llm_to_feedback()
    test_build_draft_samples_from_retrieved_results_uses_positive_and_candidate_pools()
    test_calibration_labeler_prompt_does_not_use_annoinstr()
    test_calibration_labeler_user_prompt_contains_one_candidate_only()
    test_annotate_candidates_for_calibration_uses_concurrent_runner()
    test_confirm_draft_uses_human_selected_hard_and_easy_negatives()
    test_build_positive_only_calibration_from_fewshot_uses_query_positive_pairs()
    test_positive_only_calibration_compare_requires_positive_label()
