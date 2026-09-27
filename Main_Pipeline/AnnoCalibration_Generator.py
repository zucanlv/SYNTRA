"""
Annotation calibration set generator.

This script builds a task-level calibration artifact for AnnoInstr self-refine
outside the main pipeline. It starts from golden few-shot queries, retrieves
candidate documents with the existing dense retriever, labels each retrieved
candidate with an independent high-level task prompt (not an AnnoInstr file),
and writes a human-review draft. A human then chooses one hard negative and one
easy negative per query, and this script converts the reviewed draft into the
confirmed calibration set consumed by ``main.py``.

Lifecycle:
1. Prepare golden queries from ``Few_Shot_Example``.
2. Run dense retrieval with ``Query2PassageTool`` to get candidate documents.
3. Label retrieved candidates one document per LLM call using only
   ``HighLevel_Task_Definition`` context.
4. Save a draft JSON for human selection of one hard and one easy negative.
5. Convert the human-reviewed draft into a confirmed calibration set.
6. Use the confirmed JSON in ``main.py`` test-mode AnnoInstr self-refine.

Typical usage:

Generate Query2Passage input only:

    python Main_Pipeline/AnnoCalibration_Generator.py \
      --task msmarco \
      --mode prepare-query-input \
      --output-dir Main_Pipeline/Results/Instructions/msmarco \
      --golden-query-count 20

Run dense retrieval and produce a human-review draft in one command:

    python Main_Pipeline/AnnoCalibration_Generator.py \
      --task msmarco \
      --mode run-q2p-draft \
      --faiss-dir /path/to/faiss \
      --embedding-model-path /path/to/bge-m3 \
      --top-k 50 \
      --golden-query-count 20 \
      --hard-pool-size 20 \
      --easy-pool-size 20 \
      --labeler-model gpt-4o-mini \
      --labeler-model-server https://api.openai.com/v1 \
      --labeler-max-workers 8 \
      --output-dir Main_Pipeline/Results/Instructions/msmarco

Build a draft from an existing Query2Passage output:

    python Main_Pipeline/AnnoCalibration_Generator.py \
      --task msmarco \
      --mode draft-from-q2p \
      --q2p-output /path/to/AnnoCalibrationQ2P_msmarco.json \
      --hard-pool-size 20 \
      --easy-pool-size 20 \
      --labeler-model gpt-4o-mini \
      --labeler-max-workers 8 \
      --output-path /path/to/AnnoCalibrationDraft_msmarco.json

After human review, fill these fields in each draft sample:

    "human_review": {
      "selected_hard_negative_doc_id": "...",
      "selected_easy_negative_doc_id": "...",
      "notes": "optional notes"
    }

Convert the reviewed draft to confirmed calibration format:

    python Main_Pipeline/AnnoCalibration_Generator.py \
      --task msmarco \
      --mode confirm-from-draft \
      --q2p-output /path/to/AnnoCalibrationDraft_msmarco_reviewed.json \
      --output-path /path/to/AnnoCalibration_msmarco_confirmed.json

Use the confirmed calibration set in ``main.py`` test mode:

    anno_self_refine_source: calibration_set
    anno_calibration_path: /path/to/AnnoCalibration_msmarco_confirmed.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import json5

from Few_Shot_Example import Few_Shot_Example
from HighLevel_Def import HighLevel_Task_Definition


LABEL_ALIASES = {
    "positive": "positive",
    "hard negative": "hard_negative",
    "hard_negative": "hard_negative",
    "hard-negative": "hard_negative",
    "easy negative": "easy_negative",
    "easy_negative": "easy_negative",
    "easy-negative": "easy_negative",
}


def normalize_label(label: Any) -> str:
    return LABEL_ALIASES.get(str(label or "").strip().lower(), str(label or "").strip().lower())


def _candidate(
    doc_id: Any,
    doc: str,
    *,
    expected_label: str | None = None,
    source: str,
    rank: Any = None,
    score: Any = None,
    annotator_score: Any = None,
    annotator_reasoning: str | None = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "doc_id": str(doc_id),
        "doc": doc or "",
        "source": source,
    }
    if expected_label:
        item["expected_label"] = expected_label
    if rank is not None:
        item["rank"] = rank
    if score is not None:
        item["score"] = score
    if annotator_score is not None:
        item["annotator_score"] = annotator_score
    if annotator_reasoning:
        item["annotator_reasoning"] = annotator_reasoning
    return item


def build_query2passage_input_from_fewshot(
    task: str,
    *,
    golden_query_count: int = 5,
) -> Dict[str, Any]:
    """Create a Query2Passage-compatible query batch from Few_Shot_Example."""
    examples = Few_Shot_Example.get(task)
    if not examples:
        raise ValueError(f"No Few_Shot_Example entries for task '{task}'")

    results = []
    for i, ex in enumerate(examples[:golden_query_count], 1):
        query = ex.get("query", "")
        positive = ex.get("Positive", "")
        if not query or not positive:
            continue
        query_id = f"{task}_fs_{i:03d}"
        results.append({
            "doc_id": f"{query_id}_pos",
            "doc": positive,
            "query_id": query_id,
            "queries": [{"query": query, "query_id": query_id}],
        })
    return {
        "task": task,
        "status": "query2passage_input",
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "results": results,
    }


def build_positive_only_calibration_from_fewshot(
    task: str,
    *,
    golden_query_count: int | None = None,
) -> Dict[str, Any]:
    """Create a confirmed calibration artifact from few-shot query-positive pairs.

    Unlike the full calibration workflow, this artifact intentionally contains
    no hard/easy negatives. It only verifies that AnnoInstr keeps known
    few-shot positives on the positive side of the decision boundary.
    """
    examples = Few_Shot_Example.get(task)
    if not examples:
        raise ValueError(f"No Few_Shot_Example entries for task '{task}'")

    subset = examples[:golden_query_count] if golden_query_count is not None else examples
    samples = []
    for i, ex in enumerate(subset, 1):
        query = str(ex.get("query", "")).strip()
        positive = str(ex.get("Positive", "")).strip()
        if not query or not positive:
            continue
        query_id = f"{task}_fs_pos_{i:03d}"
        samples.append({
            "query_id": query_id,
            "query": query,
            "candidates": [
                _candidate(
                    f"{query_id}_doc",
                    positive,
                    expected_label="positive",
                    source="few_shot_positive_only",
                )
            ],
        })

    if not samples:
        raise ValueError(f"No usable query/Positive few-shot pairs for task '{task}'")

    return {
        "task": task,
        "status": "confirmed",
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source": "few_shot_positive_only",
        "samples": samples,
    }


def _iter_query_results(retrieved_results: Iterable[Dict[str, Any]]):
    for item in retrieved_results:
        query_id = item.get("query_id") or str(item.get("doc_id") or "")
        positive_doc_id = str(item.get("doc_id") or f"{query_id}_pos")
        positive_doc = item.get("doc") or ""
        for q_obj in item.get("queries", []) or []:
            if isinstance(q_obj, dict) and q_obj.get("query"):
                yield query_id, q_obj.get("query"), positive_doc_id, positive_doc, q_obj


def _extract_json(text: str) -> Dict[str, Any] | None:
    try:
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()
        return json5.loads(text.strip().strip("`"))
    except Exception:
        return None


def _build_calibration_labeler_system_prompt(task: str) -> str:
    if task not in HighLevel_Task_Definition:
        raise ValueError(f"Task '{task}' not found in HighLevel_Task_Definition")
    query_type, doc_type, relevance_def = HighLevel_Task_Definition[task]
    return f"""You are an expert LLM annotator for information retrieval evaluation.

Task Context:
- Query Type: {query_type}
- Document Type: {doc_type}
- Relevance Definition: {relevance_def}

This is NOT using an existing annotation instruction. Classify exactly one candidate document for the given query with the task context and relevance definition above. Do not output a doc_id.

Use this fixed label mapping:
- score 3 or score 2: positive: regarding the query, the document satisfies the task-specific relevance definition well enough to be a valid positive.
- score 1: hard_negative: regarding the query, the document looks plausible, but fails the task-specific relevance definition.
- score 0: easy_negative: regarding the query, the document is clearly outside the task-specific relevance criteria.

Output ONLY valid JSON with this schema:
{{"score": 0-3, "classification": "positive|hard_negative|easy_negative", "reasoning": "a brief reasoning for the classification"}}.
"""


def _build_calibration_labeler_user_prompt(query: str, candidate: Dict[str, Any]) -> str:
    return (
        f"Query: {query}\n\n"
        "Candidate document:\n\n"
        f"{candidate.get('doc', '')}\n"
    )


def _normalize_single_doc_labeler_output(parsed: Dict[str, Any]) -> Dict[str, Any] | None:
    if isinstance(parsed.get("annotations"), list) and parsed["annotations"]:
        parsed = parsed["annotations"][0]
    if not isinstance(parsed, dict):
        return None
    return parsed


def annotate_one_candidate_for_calibration(
    task: str,
    query: str,
    candidate: Dict[str, Any],
    *,
    llm_cfg: Dict[str, Any],
    max_retries: int = 1,
) -> Dict[str, Any]:
    """Label one retrieved candidate with a task-context prompt, not an AnnoInstr prompt."""
    from qwen_agent.agents import Assistant

    input_doc_id = str(candidate.get("doc_id", ""))
    messages = [{"role": "user", "content": _build_calibration_labeler_user_prompt(query, candidate)}]
    labeler = Assistant(llm=llm_cfg, system_message=_build_calibration_labeler_system_prompt(task))
    raw = ""
    for attempt in range(max_retries + 1):
        for response in labeler.run(messages):
            raw = response[-1]["content"]
        parsed = _extract_json(raw)
        if parsed:
            ann = _normalize_single_doc_labeler_output(parsed)
            if ann:
                label = normalize_label(ann.get("classification"))
                if label in {"positive", "hard_negative", "easy_negative"}:
                    return {
                        "classification": label,
                        "score": ann.get("score"),
                        "reasoning": ann.get("reasoning", ""),
                    }
        if attempt < max_retries:
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "Return ONLY valid JSON for this one document with score, classification, and reasoning. Do not output doc_id.",
            })
    raise RuntimeError(f"Calibration labeler failed for doc_id={input_doc_id} query={query}")


def annotate_candidates_for_calibration(
    task: str,
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    llm_cfg: Dict[str, Any],
    max_retries: int = 1,
    concurrency_enabled: bool = True,
    max_workers: int | None = 8,
    jitter: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """Label retrieved candidates one LLM call per candidate."""
    from concurrent_runner import run_concurrent

    kwargs_list = [
        {
            "task": task,
            "query": query,
            "candidate": candidate,
            "llm_cfg": llm_cfg,
            "max_retries": max_retries,
        }
        for candidate in candidates
    ]
    results = run_concurrent(
        annotate_one_candidate_for_calibration,
        kwargs_list,
        enabled=concurrency_enabled,
        max_workers=max_workers,
        jitter=jitter,
    )
    labels: Dict[str, Dict[str, Any]] = {}
    for candidate, label in zip(candidates, results):
        labels[str(candidate.get("doc_id", ""))] = label
    return labels


def annotate_retrieved_results_for_calibration(
    task: str,
    retrieved_results: List[Dict[str, Any]],
    *,
    llm_cfg: Dict[str, Any],
    max_retries: int = 1,
    concurrency_enabled: bool = True,
    max_workers: int | None = 8,
    jitter: float = 0.0,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    labels_by_query: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for query_id, query, positive_doc_id, positive_doc, q_obj in _iter_query_results(retrieved_results):
        seen_positive_text = positive_doc.strip()
        candidates = []
        for cand in q_obj.get("candidates", []) or []:
            doc_id = str(cand.get("doc_id", ""))
            doc = cand.get("doc", "")
            if doc_id == positive_doc_id:
                continue
            if seen_positive_text and doc.strip() == seen_positive_text:
                continue
            candidates.append(cand)
        key = str(query_id or query)
        labels_by_query[key] = annotate_candidates_for_calibration(
            task,
            query,
            candidates,
            llm_cfg=llm_cfg,
            max_retries=max_retries,
            concurrency_enabled=concurrency_enabled,
            max_workers=max_workers,
            jitter=jitter,
        ) if candidates else {}
    return labels_by_query


def build_draft_samples_from_retrieved_results(
    task: str,
    retrieved_results: List[Dict[str, Any]],
    *,
    candidate_labels_by_query: Dict[str, Dict[str, Dict[str, Any]]],
    hard_pool_size: int = 20,
    easy_pool_size: int = 20,
) -> Dict[str, Any]:
    """Build a human-review draft using LLM-labeled candidate pools."""
    samples = []
    for idx, (query_id, query, positive_doc_id, positive_doc, q_obj) in enumerate(
        _iter_query_results(retrieved_results), 1
    ):
        seen_positive_text = positive_doc.strip()
        labels = candidate_labels_by_query.get(str(query_id or query), {})
        hard_pool = []
        easy_pool = []
        for cand in q_obj.get("candidates", []) or []:
            doc_id = str(cand.get("doc_id", ""))
            doc = cand.get("doc", "")
            if doc_id == positive_doc_id:
                continue
            if seen_positive_text and doc.strip() == seen_positive_text:
                continue
            label_info = labels.get(doc_id)
            if not label_info:
                continue
            label = normalize_label(label_info.get("classification"))
            item = _candidate(
                doc_id,
                doc,
                source="calibration_labeler",
                rank=cand.get("rank"),
                score=cand.get("score"),
                annotator_score=label_info.get("score"),
                annotator_reasoning=label_info.get("reasoning"),
            )
            if label == "hard_negative" and len(hard_pool) < hard_pool_size:
                hard_pool.append(item)
            elif label == "easy_negative" and len(easy_pool) < easy_pool_size:
                easy_pool.append(item)

        sample_id = query_id or f"{task}_calib_{idx:03d}"
        samples.append({
            "query_id": sample_id,
            "query": query,
            "positive": _candidate(
                positive_doc_id,
                positive_doc,
                expected_label="positive",
                source="few_shot_positive",
            ),
            "hard_negative_candidates": hard_pool,
            "easy_negative_candidates": easy_pool,
            "human_review": {
                "selected_hard_negative_doc_id": "",
                "selected_easy_negative_doc_id": "",
                "notes": "",
            },
        })

    return {
        "task": task,
        "status": "draft",
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "instructions": (
            "Candidates were pre-labeled by a calibration labeler prompt built from the high-level "
            "task definition, not from an AnnoInstr file. For each sample, choose one doc_id from "
            "hard_negative_candidates and one doc_id from easy_negative_candidates after human review."
        ),
        "samples": samples,
    }




def _select_candidate(candidates: List[Dict[str, Any]], doc_id: str, label: str) -> Dict[str, Any]:
    for cand in candidates:
        if str(cand.get("doc_id")) == str(doc_id):
            selected = dict(cand)
            selected["doc_id"] = str(selected.get("doc_id", ""))
            selected["expected_label"] = label
            selected["source"] = selected.get("source") or "human_selected_dense_retrieval"
            return selected
    raise ValueError(f"Selected {label} doc_id not found in candidate pool: {doc_id}")


def confirm_draft_from_human_selection(draft: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a human-reviewed draft into the confirmed calibration schema."""
    confirmed_samples = []
    for sample in draft.get("samples", []):
        review = sample.get("human_review") or {}
        hard_id = review.get("selected_hard_negative_doc_id")
        easy_id = review.get("selected_easy_negative_doc_id")
        if not hard_id or not easy_id:
            raise ValueError(
                f"Sample {sample.get('query_id') or sample.get('query')} is missing human selections"
            )

        positive = dict(sample.get("positive") or {})
        if not positive.get("doc_id") or not positive.get("doc"):
            raise ValueError(f"Sample {sample.get('query_id')} missing positive document")
        positive["doc_id"] = str(positive["doc_id"])
        positive["expected_label"] = "positive"
        positive["source"] = positive.get("source") or "few_shot_positive"

        hard = _select_candidate(
            sample.get("hard_negative_candidates", []),
            str(hard_id),
            "hard_negative",
        )
        easy = _select_candidate(
            sample.get("easy_negative_candidates", []),
            str(easy_id),
            "easy_negative",
        )
        confirmed_samples.append({
            "query_id": sample.get("query_id") or sample.get("query"),
            "query": sample["query"],
            "candidates": [positive, hard, easy],
            "human_review_notes": review.get("notes", ""),
        })

    return {
        "task": draft.get("task", ""),
        "status": "confirmed",
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source_draft_created_at": draft.get("created_at", ""),
        "samples": confirmed_samples,
    }

def load_anno_calibration(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Calibration file has no non-empty samples list: {path}")
    for sample in samples:
        if not sample.get("query"):
            raise ValueError(f"Calibration sample missing query: {sample}")
        candidates = sample.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Calibration sample missing candidates: {sample.get('query_id')}")
        for cand in candidates:
            label = normalize_label(cand.get("expected_label"))
            if label not in {"positive", "hard_negative", "easy_negative"}:
                raise ValueError(
                    f"Candidate {cand.get('doc_id')} has invalid expected_label: "
                    f"{cand.get('expected_label')}"
                )
            cand["expected_label"] = label
            cand["doc_id"] = str(cand.get("doc_id", ""))
    return data


def _label_for_calibration_compare(label: Any, label_granularity: str = "three_way") -> str:
    normalized = normalize_label(label)
    if label_granularity == "posneg":
        if normalized == "positive":
            return "positive"
        if normalized in {"hard_negative", "easy_negative"}:
            return "negative"
    return normalized


def compare_annotations_against_calibration(
    annotation_samples: List[Dict[str, Any]],
    calibration_samples: List[Dict[str, Any]],
    *,
    label_granularity: str = "three_way",
) -> Tuple[bool, str, Dict[str, Any]]:
    """Compare annotator output against expected labels from confirmed calibration.

    label_granularity="three_way" requires exact positive/hard/easy labels.
    label_granularity="posneg" treats hard_negative and easy_negative as one
    negative class, so either predicted negative subtype matches either expected
    negative subtype.
    """
    if label_granularity not in {"three_way", "posneg"}:
        raise ValueError(f"Unsupported label_granularity: {label_granularity}")
    expected_by_query: Dict[str, Dict[str, Dict[str, Any]]] = {
        str(s.get("query_id") or s.get("query")): {
            str(c.get("doc_id")): c
            for c in s.get("candidates", [])
        }
        for s in calibration_samples
    }

    mismatches = []
    total = 0
    correct = 0
    for sample in annotation_samples:
        key = str(sample.get("query_id") or sample.get("query"))
        expected = expected_by_query.get(key, {})
        predicted = {
            str(a.get("doc_id")): a
            for a in sample.get("annotations", [])
        }
        for doc_id, expected_candidate in expected.items():
            expected_label = normalize_label(expected_candidate.get("expected_label"))
            expected_compare_label = _label_for_calibration_compare(
                expected_label, label_granularity
            )
            total += 1
            predicted_annotation = predicted.get(doc_id)
            pred_label = (
                normalize_label(predicted_annotation.get("classification"))
                if predicted_annotation
                else "missing"
            )
            pred_compare_label = _label_for_calibration_compare(
                pred_label, label_granularity
            )
            if pred_compare_label == expected_compare_label:
                correct += 1
                continue
            mismatches.append({
                "query_id": key,
                "query": sample.get("query", ""),
                "doc_id": doc_id,
                "expected": expected_compare_label,
                "predicted": pred_compare_label,
                "expected_original": expected_label,
                "predicted_original": pred_label,
                "doc": expected_candidate.get("doc", ""),
                "expected_reasoning": (
                    expected_candidate.get("reasoning")
                    or expected_candidate.get("annotator_reasoning", "")
                ),
                "predicted_reasoning": (
                    predicted_annotation.get("reasoning", "")
                    if predicted_annotation
                    else ""
                ),
            })

    passed = total > 0 and not mismatches
    if passed:
        feedback = f"All {total} calibration labels matched expected labels."
    else:
        lines = [
            f"{len(mismatches)} calibration mismatches out of {total} labels.",
        ]
        for m in mismatches:
            lines.append(
                f"- query_id={m['query_id']} doc_id={m['doc_id']}: "
                f"expected {m['expected']}, got {m['predicted']}"
            )
        feedback = "\n".join(lines)

    return passed, feedback, {
        "total": total,
        "correct": correct,
        "label_granularity": label_granularity,
        "mismatches": mismatches,
    }


def build_calibration_refinement_feedback_prompt(
    annotation_instruction: str,
    task_name: str,
    compare_details: Dict[str, Any],
) -> str:
    """Build an LLM prompt that turns deterministic calibration errors into guidance."""
    query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]
    mismatches = compare_details.get("mismatches", [])
    mismatches_json = json.dumps(mismatches, ensure_ascii=False, indent=2)

    return f"""You are a senior prompt evaluator for an information retrieval annotation pipeline.

## Your Task
The calibration labels are already confirmed, and pass/fail has already been decided by exact label comparison. Do not re-grade the calibration set and do not decide whether the run passed.

Your job is to turn the confirmed misclassifications below into constructive feedback for rewriting the annotation instruction. The feedback must explain what the current instruction is causing the annotator to misunderstand and how the next instruction should prevent the same mistakes.

## Step-by-step guideline
1. **Understand the task boundary.** Read the query type, document type, and relevance definition. Identify what a document must actually do to count as positive for this task.
2. **Analyze each mismatch.** For every mismatch, compare the expected label with the predicted label. Use the query, document text, expected reasoning, and predicted reasoning to infer the likely failure mode.
3. **Group repeated failure patterns.** Infer the recurring failure modes from the mismatches themselves instead of forcing them into predefined error categories.
4. **Diagnose instruction gaps.** Explain which missing or weak parts of the current annotation instruction likely allowed these errors.
5. **Write actionable refinement guidance.** Suggest concrete wording-level improvements for the next instruction. Focus on task-specific relevance criteria and hard-negative boundaries, not generic advice like "be more careful".

## Task Context
- Query Type: {query_type}
- Document Type: {doc_type}
- Relevance Definition: {relevance_def}

## Current Annotation Instruction
{annotation_instruction}

## Deterministic Calibration Result
- Total labels: {compare_details.get("total", 0)}
- Correct labels: {compare_details.get("correct", 0)}
- Mismatches: {len(mismatches)}

## Confirmed Misclassifications
Each item contains the query/document pair, the confirmed expected label, the annotator's predicted label, and any available expected/predicted reasoning.
{mismatches_json}

## Feedback Goal
Write structured feedback that guides the instruction refiner's thinking. The fields below are not separate checkboxes; together they should form a coherent diagnosis of the global failure pattern.

Do not merely enumerate individual errors. Use specific mismatches as evidence, but convert them into generalizable guidance that can prevent the same class of mistakes on future query-document pairs. Avoid generic advice such as "be more careful" or "think harder".

## Output Format (MANDATORY — valid JSON only, no Markdown)
Output a single JSON object:
{{
  "feedback": "Concise global summary of the main failure pattern. Cite representative mismatches only as evidence for the broader diagnosis.",
  "positive_issues": "Global issues involving positive classifications, especially cases where plausibility caused false positives; use 'none' if not relevant.",
  "negative_issues": "Global issues involving hard/easy negative distinctions or missed hard-negative boundaries; use 'none' if not relevant.",
  "instruction_gaps": "The missing, weak, or ambiguous instruction guidance that likely caused the observed error pattern.",
  "suggestions": "Concrete task-specific changes the next instruction should make to prevent this class of errors."
}}"""

def generate_calibration_refinement_feedback(
    annotation_instruction: str,
    task_name: str,
    llm_cfg: dict,
    compare_feedback: str,
    compare_details: Dict[str, Any],
    logger=None,
) -> str:
    """Ask an LLM to convert calibration mismatches into actionable refine feedback.

    The LLM does not decide pass/fail; deterministic label comparison remains
    authoritative. On parse or model failure, this falls back to the raw
    calibration mismatch report.
    """
    if not compare_details.get("mismatches"):
        return compare_feedback

    prompt = build_calibration_refinement_feedback_prompt(
        annotation_instruction=annotation_instruction,
        task_name=task_name,
        compare_details=compare_details,
    )

    try:
        from qwen_agent.agents import Assistant

        evaluator = Assistant(llm=llm_cfg)
        raw = ""
        for response in evaluator.run([{"role": "user", "content": prompt}]):
            raw = response[-1]["content"]

        parsed = json5.loads(raw)
        parts = [compare_feedback]
        if parsed.get("feedback"):
            parts.append(parsed["feedback"])
        if parsed.get("positive_issues") and parsed["positive_issues"].lower() != "none":
            parts.append(f"Positive issues: {parsed['positive_issues']}")
        if parsed.get("negative_issues") and parsed["negative_issues"].lower() != "none":
            parts.append(f"Negative issues: {parsed['negative_issues']}")
        if parsed.get("instruction_gaps") and parsed["instruction_gaps"].lower() != "none":
            parts.append(f"Instruction gaps: {parsed['instruction_gaps']}")
        if parsed.get("suggestions"):
            parts.append(f"Suggestions: {parsed['suggestions']}")
        return "\n".join(parts)
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "Failed to generate LLM calibration feedback: %s; using raw mismatch report",
                exc,
            )
        return compare_feedback


def write_json(data: Dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out


def _default_output(prefix: str, task: str, output_dir: str | Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"{prefix}_{task}_{ts}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate annotation calibration draft files outside the main pipeline."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--mode",
        choices=["prepare-query-input", "draft-from-q2p", "confirm-from-draft", "run-q2p-draft"],
        default="prepare-query-input",
    )
    parser.add_argument("--output-dir", default="Results/Instructions")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--q2p-input", default=None)
    parser.add_argument("--q2p-output", default=None)
    parser.add_argument("--golden-query-count", type=int, default=20)
    parser.add_argument("--hard-pool-size", type=int, default=20)
    parser.add_argument("--easy-pool-size", type=int, default=20)
    parser.add_argument("--labeler-model", default="gpt-4o-mini")
    parser.add_argument("--labeler-model-server", default=None)
    parser.add_argument("--labeler-api-key", default=None)
    parser.add_argument("--labeler-concurrency-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--labeler-max-workers", type=int, default=8)
    parser.add_argument("--labeler-jitter", type=float, default=0.0)

    # Query2Passage parameters for run-q2p-draft.
    parser.add_argument("--faiss-dir", default=None)
    parser.add_argument("--embedding-model-name", default="BAAI/bge-m3")
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--faiss-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-ids", default=None)
    args = parser.parse_args()

    if args.mode == "prepare-query-input":
        data = build_query2passage_input_from_fewshot(
            args.task, golden_query_count=args.golden_query_count
        )
        out = Path(args.output_path) if args.output_path else _default_output(
            "AnnoCalibrationQueries", args.task, args.output_dir
        )
        write_json(data, out)
        print(out)
        return

    if args.mode == "draft-from-q2p":
        if not args.q2p_output:
            raise SystemExit("--q2p-output is required for draft-from-q2p")
        with open(args.q2p_output, "r", encoding="utf-8") as f:
            q2p = json.load(f)
        llm_cfg = {
            "model": args.labeler_model,
            "model_server": args.labeler_model_server or os.getenv("OPENAI_BASE_URL"),
            "api_key": args.labeler_api_key or os.getenv("OPENAI_API_KEY"),
        }
        labels = annotate_retrieved_results_for_calibration(
            args.task,
            q2p.get("results", []),
            llm_cfg=llm_cfg,
            concurrency_enabled=args.labeler_concurrency_enabled,
            max_workers=args.labeler_max_workers,
            jitter=args.labeler_jitter,
        )
        draft = build_draft_samples_from_retrieved_results(
            args.task,
            q2p.get("results", []),
            candidate_labels_by_query=labels,
            hard_pool_size=args.hard_pool_size,
            easy_pool_size=args.easy_pool_size,
        )
        out = Path(args.output_path) if args.output_path else _default_output(
            "AnnoCalibrationDraft", args.task, args.output_dir
        )
        write_json(draft, out)
        print(out)
        return

    if args.mode == "confirm-from-draft":
        if not args.q2p_output:
            raise SystemExit("--q2p-output must point to the human-reviewed draft for confirm-from-draft")
        with open(args.q2p_output, "r", encoding="utf-8") as f:
            draft = json.load(f)
        confirmed = confirm_draft_from_human_selection(draft)
        out = Path(args.output_path) if args.output_path else Path(args.output_dir) / f"AnnoCalibration_{args.task}_confirmed.json"
        write_json(confirmed, out)
        print(out)
        return

    if args.mode == "run-q2p-draft":
        if not args.faiss_dir:
            raise SystemExit("--faiss-dir is required for run-q2p-draft")
        from Query2Passge import Query2PassageTool

        q2p_input = Path(args.q2p_input) if args.q2p_input else _default_output(
            "AnnoCalibrationQueries", args.task, args.output_dir
        )
        if not args.q2p_input:
            write_json(
                build_query2passage_input_from_fewshot(
                    args.task, golden_query_count=args.golden_query_count
                ),
                q2p_input,
            )

        q2p_output = Path(args.q2p_output) if args.q2p_output else _default_output(
            "AnnoCalibrationQ2P", args.task, args.output_dir
        )
        params = {
            "query_file": str(q2p_input),
            "faiss_dir": args.faiss_dir,
            "top_k": args.top_k,
            "output_file": str(q2p_output),
            "model_name": args.embedding_model_name,
            "model_path": args.embedding_model_path,
            "batch_size": args.batch_size,
            "device": args.device,
            "faiss_gpu": args.faiss_gpu,
            "inject_origin_score": False,
        }
        if args.gpu_ids:
            params["gpu_ids"] = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
        result = json5.loads(Query2PassageTool().call(json5.dumps(params)))
        if result.get("status") != "success":
            raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))

        with open(q2p_output, "r", encoding="utf-8") as f:
            q2p = json.load(f)
        llm_cfg = {
            "model": args.labeler_model,
            "model_server": args.labeler_model_server or os.getenv("OPENAI_BASE_URL"),
            "api_key": args.labeler_api_key or os.getenv("OPENAI_API_KEY"),
        }
        labels = annotate_retrieved_results_for_calibration(
            args.task,
            q2p.get("results", []),
            llm_cfg=llm_cfg,
            concurrency_enabled=args.labeler_concurrency_enabled,
            max_workers=args.labeler_max_workers,
            jitter=args.labeler_jitter,
        )
        draft = build_draft_samples_from_retrieved_results(
            args.task,
            q2p.get("results", []),
            candidate_labels_by_query=labels,
            hard_pool_size=args.hard_pool_size,
            easy_pool_size=args.easy_pool_size,
        )
        out = Path(args.output_path) if args.output_path else _default_output(
            "AnnoCalibrationDraft", args.task, args.output_dir
        )
        write_json(draft, out)
        print(out)


if __name__ == "__main__":
    main()
