import json
import os
import json5
import logging
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from HighLevel_Def import HighLevel_Task_Definition
from datetime import datetime

os.makedirs("Logs", exist_ok=True)

# Configure logging to write to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"Logs/AnnoInstr_Generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

ANNOTATION_PROMPT_MODE_MULTI_DOC_WITH_DOC_ID = "multi_doc_with_doc_id"
ANNOTATION_PROMPT_MODE_SINGLE_DOC_NO_DOC_ID = "single_doc_no_doc_id"
VALID_ANNOTATION_PROMPT_MODES = {
    ANNOTATION_PROMPT_MODE_MULTI_DOC_WITH_DOC_ID,
    ANNOTATION_PROMPT_MODE_SINGLE_DOC_NO_DOC_ID,
}


def normalize_annotation_prompt_mode(value: Any) -> str:
    mode = str(value or ANNOTATION_PROMPT_MODE_MULTI_DOC_WITH_DOC_ID).strip()
    if mode not in VALID_ANNOTATION_PROMPT_MODES:
        raise ValueError(
            f"Invalid annotation_prompt_mode={mode!r}; expected one of "
            f"{sorted(VALID_ANNOTATION_PROMPT_MODES)}"
        )
    return mode


def build_anno_instr_meta_prompt(
    query_type: str,
    doc_type: str,
    relevance_def: str,
    annotation_prompt_mode: str,
) -> str:
    common = f"""
You are an expert LLM annotator for information retrieval evaluation. Create a concise system instruction for another LLM annotator to evaluate documents and classify them into three categories: positive, hard negative, or easy negative.

**Critical thinking guideline:** The instruction should push the annotator to reason from the specific query-document relationship in this task: identify what would truly satisfy the query, what only looks relevant, and why the candidate crosses or fails that boundary. Avoid generic relevance advice.

The system instruction should specify following parts clearly:
**1. Task Context:**
- Query Type: {query_type}
- Document Type: {doc_type}
- Relevance Definition: {relevance_def}

**2. Scoring & Classification System:**
The instruction must define a task-specific 0-3 absolute scoring rubric grounded in the Query Type, Document Type, and Relevance Definition above. Guide the another annotator to reason about whether each document satisfies the task's stated relevance criteria, and to distinguish strong satisfaction, valid-but-less-complete positive cases, related-but-not-relevant cases, and clearly irrelevant cases.

Keep this label mapping fixed:
- **Score 3 -> positive**: Strongly satisfies the task-specific relevance definition.
- **Score 2 -> positive**: Satisfies the task-specific relevance definition well enough to be a valid positive, but is not enough to be a score 3 document.
- **Score 1 -> hard_negative**: The document looks plausible for the query in this task, but fails the task-specific relevance definition.
- **Score 0 -> easy_negative**: Clearly unrelated or outside the task-specific relevance criteria.

**3. Hard Negative Guidance:** The instruction should explain hard negatives with respect to the task’s actual relevance criteria: they may appear plausible for this task, but they fail the definition of relevance.
"""
    if annotation_prompt_mode == ANNOTATION_PROMPT_MODE_SINGLE_DOC_NO_DOC_ID:
        contract = """
**4. Input Format:**
- One query string
- Exactly one candidate document

**5. Output Format (valid JSON only):**
{"score": 0-3, "classification": "positive|hard_negative|easy_negative", "reasoning": "brief"}

The instruction should be concise, emphasize hard negative identification, and require valid JSON output with only score, classification, and reasoning.
Output ONLY the system instruction text. **DO NOT** wrap the instruction in a JSON object!
"""
    else:
        contract = """
**4. Input Format:**
- One query string
- List of candidate documents with identifiers (doc_id)

**5. Annotation Protocol:**
1. Process sequentially: Annotate each document exactly once in the order provided. Ensure no doc_id is duplicated or omitted in the final output.
2. Independent evaluation: Evaluate each document in isolation based strictly on its relationship with the query. Do not allow the quality or content of other documents in the list to influence the current score (avoid relative ranking; use absolute scoring).

**6. Output Format (valid JSON only):**
{"query": "original query",
 "annotations": [{"doc_id": "id", "score": 0-3, "classification": "positive|hard negative|easy negative", "reasoning": "brief"}],
 "positive_doc_ids": [...], "hard_negative_doc_ids": [...], "easy_negative_doc_ids": [...]}

The instruction should be concise, emphasize hard negative identification, and require valid JSON output.
Output ONLY the system instruction text. **DO NOT** wrap the instruction in a JSON object!
"""
    return common + contract

@register_tool('annotation_instruction_generator')
class AnnotationInstructionGenerator(BaseTool):
    description = "Generate a detailed system instruction for an LLM to evaluate multiple candidate documents for a given query, scoring each and classifying them as positive, hard negative, or easy negative."
    parameters = [
        {
            'name': 'task_name',
            'type': 'string',
            'description': 'The name of the task (e.g., "MSMarco") from the high-level definitions.',
            'required': True
        },
        {
            'name': 'annotation_prompt_mode',
            'type': 'string',
            'description': 'Annotation prompt contract: multi_doc_with_doc_id or single_doc_no_doc_id.',
            'required': False
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        params_dict = json5.loads(params)
        task_name = params_dict.get('task_name')
        force_generate = params_dict.get('force_generate', False)
        try:
            annotation_prompt_mode = normalize_annotation_prompt_mode(
                params_dict.get('annotation_prompt_mode')
            )
        except ValueError as exc:
            return json5.dumps({'error': str(exc)}, ensure_ascii=False)
        
        if task_name not in HighLevel_Task_Definition:
            return json5.dumps({'error': f"Task '{task_name}' not found in HighLevel_Task_Definition."}, ensure_ascii=False)
        
        query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]
        
        # 1. Check cache
        # Generate hash from task definition to identify instruction version
        def_content = f"{query_type}|{doc_type}|{relevance_def}|{annotation_prompt_mode}"
        def_hash = hashlib.md5(def_content.encode('utf-8')).hexdigest()[:8]
        
        cache_dir = Path(__file__).parent / "Results" / "Instructions"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the latest cache file matching the hash
        cache_files = sorted(list(cache_dir.glob(f"AnnoInstr_{task_name}_{def_hash}_*.txt")), reverse=True)
        
        if cache_files and not force_generate:
            cached_path = cache_files[0]
            logging.info(f"Using cached instruction for task '{task_name}': {cached_path}")
            with open(cached_path, 'r', encoding='utf-8') as f:
                instruction_output = f.read()
            
            return json5.dumps({
                'task_name': task_name,
                'generated_instruction': instruction_output,
                'cache_path': str(cached_path),
                'from_cache': True,
                'annotation_prompt_mode': annotation_prompt_mode
            }, ensure_ascii=False)

        # 2. If no cache or force_generate, call LLM
        # Define the meta-prompt for the Annotation Quality Assessment Instruction Designer
        # Annotation logic: input one query + its candidate documents -> LLM scores each document and classifies as Positive / Hard Negative / Easy Negative
        meta_prompt = build_anno_instr_meta_prompt(
            query_type=query_type,
            doc_type=doc_type,
            relevance_def=relevance_def,
            annotation_prompt_mode=annotation_prompt_mode,
        )
        # Internal agent to generate the evaluation instruction
        llm_cfg = kwargs.get('llm_cfg', {
            'model': 'gpt-4o-mini',
            'model_server': os.getenv('OPENAI_BASE_URL'),
            'api_key': os.getenv('OPENAI_API_KEY'),
        })

        designer = Assistant(llm=llm_cfg)
        
        instruction_output = ""
        for response in designer.run([{'role': 'user', 'content': meta_prompt}]):
            instruction_output = response[-1]['content']
        
        # 3. Static validation
        ok, errors = validate_instruction(
            instruction_output, annotation_prompt_mode=annotation_prompt_mode
        )
        if not ok:
            logging.warning(f"Generated instruction failed validation: {errors}")

        # 4. Save to cache
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = cache_dir / f"AnnoInstr_{task_name}_{def_hash}_{timestamp}.txt"
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(instruction_output)
        logging.info(f"Generated instruction saved to: {save_path}")
            
        return json5.dumps({
            'task_name': task_name,
            'generated_instruction': instruction_output,
            'cache_path': str(save_path),
            'from_cache': False,
            'annotation_prompt_mode': annotation_prompt_mode
        }, ensure_ascii=False)


def validate_instruction(
    instruction: str,
    annotation_prompt_mode: str = ANNOTATION_PROMPT_MODE_MULTI_DOC_WITH_DOC_ID,
) -> tuple[bool, list[str]]:
    """
    Statically validate the generated annotation instruction for the selected
    annotation prompt contract.
    """
    annotation_prompt_mode = normalize_annotation_prompt_mode(annotation_prompt_mode)
    errors = []

    if annotation_prompt_mode == ANNOTATION_PROMPT_MODE_SINGLE_DOC_NO_DOC_ID:
        required_fields = ["score", "classification", "reasoning"]
        for field in required_fields:
            if field not in instruction:
                errors.append(f"Missing required JSON field '{field}' in instruction output format.")
    else:
        if "doc_id" not in instruction.lower():
            errors.append("Missing 'doc_id' in instruction (essential for data alignment).")
        required_fields = [
            "annotations", "score", "classification",
            "positive_doc_ids", "hard_negative_doc_ids", "easy_negative_doc_ids",
        ]
        for field in required_fields:
            if field not in instruction:
                errors.append(f"Missing required JSON field '{field}' in instruction output format.")

    if "0-3" not in instruction and not ("0" in instruction and "3" in instruction):
         errors.append("Scoring system (0-3) might be missing or incorrectly defined.")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Self-refine: evaluate & refine annotation instructions
# ---------------------------------------------------------------------------


def build_anno_instr_evaluate_prompt(
    annotation_instruction: str,
    annotation_samples: List[Dict[str, Any]],
    task_name: str,
    annotation_prompt_mode: str = ANNOTATION_PROMPT_MODE_MULTI_DOC_WITH_DOC_ID,
) -> str:
    """Build a prompt asking the LLM to evaluate annotation quality.

    Parameters
    ----------
    annotation_samples : list[dict]
        Each dict has: query, candidates (list with doc_id/doc),
        annotations (list with doc_id/classification),
        positive_doc_ids, hard_negative_doc_ids, easy_negative_doc_ids.
    """
    query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]

    samples_json = json.dumps(annotation_samples, ensure_ascii=False, indent=2)

    return f"""You are a senior quality evaluator for an information retrieval annotation pipeline.

## Your Task
Evaluate whether the **annotation results** below are correct and whether the annotation instruction is providing sufficient guidance to produce accurate positive/negative classifications.

## Step-by-step guideline:
1. **Understand the task.** Be clear on what good annotations look like in this setting according to the relevance definition.
2. **Evaluate each annotation sample in turn.** For every (query, documents, annotations) triple in the samples:
    i. **Check against the relevance definition.** For that query, do the labeled positives truly satisfy "{relevance_def}"? Are hard_negatives correctly “related but not relevant,” and easy_negatives clearly off-topic?
    ii. **Apply the evaluation criteria.** Judge positive accuracy, hard/easy negative accuracy, and consistency across samples. Note concrete misclassifications (e.g., "Sample 2: doc X labeled positive but is off-topic").
    Keep a clear per-sample record of misclassifications and instruction-related confusions for Step 3.
3. **Summarize.** From the per-sample issues, distill the main annotation quality problems and any gaps in the current annotation instruction. This summary will be used to refine the annotation instruction.
4. **Output.** Respond strictly in the required JSON output format.

## Task Context
- Query Type: {query_type}
- Document Type: {doc_type}
- Relevance Definition: {relevance_def}

## Current Annotation Instruction (used to produce the annotations below)
{annotation_instruction}

## Annotation Samples
{samples_json}

## Evaluation Criteria (check each rigorously)
1. **Positive Accuracy**: For each query, are the documents classified as "positive" truly relevant according to the relevance definition? Would a domain expert agree?
2. **Hard Negative Accuracy**: Are "hard_negative" documents genuinely related-but-not-relevant (sharing topic/keywords but failing the relevance criteria)? Or are some actually positive?
3. **Easy Negative Accuracy**: Are "easy_negative" documents truly off-topic?
4. **Classification Consistency**: Is the boundary between positive and hard_negative applied consistently across queries?
5. **Instruction Completeness**: Does the annotation instruction provide enough task-specific guidance, or is it too generic?

## Scoring Rubric (assign exactly one integer score 0–3)
> **Important**: The parenthetical examples (e.g., ...) in each score level are purely illustrative of the *type* of issue at that severity. Do not treat them as a checklist — base your score on the holistic pattern across **all five evaluation criteria above**, not just the dimensions that happen to be mentioned in the examples.

- **Score 3 — Production-Ready**: Zero misclassifications. All positives strictly satisfy the relevance definition, all hard negatives are genuinely related-but-not-relevant, all easy negatives are unambiguously off-topic. The positive/hard_negative boundary is applied consistently across every sample. The annotation instruction is also complete and task-specific — it provides sufficient guidance to reliably reproduce this quality on future batches.
- **Score 2 — Instruction Incomplete**: Zero misclassifications in this batch — every label is correct. However, the annotation instruction has minor gaps (e.g., edge cases or boundary scenarios not explicitly covered) that could cause errors on future batches. The output itself is usable for training data, but the instruction warrants a small targeted improvement.
- **Score 1 — Needs Revision**: Any misclassification is present — e.g., a hard negative that should be positive, an inconsistently applied positive/hard_negative boundary, or an easy negative that is actually on-topic. Even a single clear misclassification (not a genuinely ambiguous borderline case) warrants this score. The instruction lacks the task-specific criteria needed to prevent these errors.
- **Score 0 — Unusable**: Systematic, pervasive misclassification — the majority of labels are wrong, or the annotation instruction is so generic that it provides no task-specific criteria at all. The output cannot be used for training data without full re-annotation.

## Output Format (MANDATORY — valid JSON only, no Markdown)
Output a single JSON object:
{{
  "overall_score": 0-3,
  "passed": true if overall_score >= 3 else false,
  "feedback": "Concise summary of annotation quality issues. Cite specific query-document pairs that are misclassified and explain WHY.",
  "positive_issues": "Issues with positive classifications, or 'none'.",
  "negative_issues": "Issues with hard/easy negative classifications, or 'none'.",
  "instruction_gaps": "What the annotation instruction is missing or getting wrong, or 'none'.",
  "suggestions": "Concrete, actionable suggestions for improving the annotation instruction."
}}

IMPORTANT: A "passed" verdict (overall_score >= 3) requires **zero misclassifications AND a complete, task-specific instruction** in this batch. Any label you judge to be clearly wrong — even a single one — must result in score <= 1. Only genuinely unresolvable borderline cases (where the relevance definition itself is ambiguous) may be noted without penalizing the score."""


def evaluate_annotation_instruction(
    annotation_instruction: str,
    annotation_samples: List[Dict[str, Any]],
    task_name: str,
    llm_cfg: dict,
    logger=None,
    annotation_prompt_mode: str = ANNOTATION_PROMPT_MODE_MULTI_DOC_WITH_DOC_ID,
) -> tuple:
    """Evaluate annotation results and instruction quality.

    Returns ``(passed: bool, feedback: str)``.
    """
    log = logger or logging.getLogger(__name__)
    prompt = build_anno_instr_evaluate_prompt(
        annotation_instruction, annotation_samples, task_name,
        annotation_prompt_mode=annotation_prompt_mode,
    )

    evaluator = Assistant(llm=llm_cfg)
    raw = ""
    for response in evaluator.run([{"role": "user", "content": prompt}]):
        raw = response[-1]["content"]

    eval_details: dict = {}
    try:
        parsed = json5.loads(raw)
        score = parsed.get("overall_score", 0)
        passed = score >= 3
        parts = []
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
        feedback = "\n".join(parts)
        eval_details = {
            "score": score,
            "feedback": parsed.get("feedback", ""),
            "positive_issues": parsed.get("positive_issues", ""),
            "negative_issues": parsed.get("negative_issues", ""),
            "instruction_gaps": parsed.get("instruction_gaps", ""),
            "suggestions": parsed.get("suggestions", ""),
        }
        log.info("Annotation instruction evaluation: passed=%s score=%s", passed, score)
    except Exception as e:
        log.warning("Failed to parse evaluation JSON: %s — treating as not passed", e)
        passed = False
        feedback = f"Evaluation parse error. Raw output:\n{raw[:1000]}"

    return passed, feedback, eval_details


def build_anno_instr_refine_prompt(
    annotation_instruction: str,
    annotation_samples: List[Dict[str, Any]],
    feedback: str,
    task_name: str,
    annotation_prompt_mode: str = ANNOTATION_PROMPT_MODE_MULTI_DOC_WITH_DOC_ID,
    include_annotation_samples: bool = True,
) -> str:
    """Build a prompt asking the LLM to rewrite the annotation instruction."""
    query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]
    annotation_prompt_mode = normalize_annotation_prompt_mode(annotation_prompt_mode)
    sample_section = ""
    if include_annotation_samples:
        samples_brief = json.dumps(annotation_samples, ensure_ascii=False, indent=2)
        sample_section = f"""
### Sample Annotations (from the current instruction)
{samples_brief}
"""

    if annotation_prompt_mode == ANNOTATION_PROMPT_MODE_SINGLE_DOC_NO_DOC_ID:
        return f"""You are a prompt engineering expert specializing in information retrieval evaluation.

## Context
We have an annotation instruction that guides an LLM to evaluate exactly one candidate document for one query and classify that document as positive, hard_negative, or easy_negative.

- Query Type: {query_type}
- Document Type: {doc_type}
- Relevance Definition: {relevance_def}

## Problem
The current instruction was evaluated and received the following feedback:

### Evaluation Feedback
{feedback}
{sample_section}
### Current Instruction
{annotation_instruction}

## Your Task
Rewrite the annotation instruction to fix all issues identified in the feedback. The rewritten instruction must produce more accurate and consistent classifications for the single query-document annotation setting.

## Requirements
1. **Preserve the scoring system**: Keep the 0-3 score and three-class classification (positive/hard_negative/easy_negative).
2. **Preserve the output format**: Output only one JSON object with score, classification, and reasoning.
3. **Preserve what works**: If parts of the current instruction are effective, keep them.
4. **Fix what's broken**: Address every issue in the feedback, especially misclassification patterns and weak relevance-boundary guidance.
5. **Add task-specific guidance**: Include concrete criteria specific to {query_type}/{doc_type} that help distinguish true positives from hard negatives.
6. **Be precise about boundaries**: Clearly define where positive ends and hard_negative begins for this task.
7. **Preserve the instruction style**: Keep the same style and structure as the current annotation instruction above (e.g. same level of detail, sectioning, and bullet conventions).

## Output Format
Output ONLY the rewritten annotation instruction (plain text). No JSON wrapping, no Markdown fences. This will be used as a system prompt for the annotator LLM."""

    return f"""You are a prompt engineering expert specializing in information retrieval evaluation.

## Context
We have an annotation instruction that guides an LLM to evaluate one or more candidate documents for a query and classify each document as positive, hard_negative, or easy_negative.

- Query Type: {query_type}
- Document Type: {doc_type}
- Relevance Definition: {relevance_def}
- Annotation Prompt Contract: {annotation_prompt_mode}

## Problem
The current instruction was evaluated and received the following feedback:

### Evaluation Feedback
{feedback}
{sample_section}
### Current Instruction
{annotation_instruction}

## Your Task
Rewrite the annotation instruction to fix all issues identified in the feedback. The rewritten instruction must produce more accurate and consistent classifications for the multi-document annotation setting.

## Requirements
1. **Preserve the scoring system**: Keep the 0-3 score and three-class classification (positive/hard_negative/easy_negative).
2. **Preserve the multi-doc output format**: Keep the JSON schema with annotations, positive_doc_ids, hard_negative_doc_ids, and easy_negative_doc_ids, and require exact doc_id echoing for each candidate.
3. **Preserve what works**: If parts of the current instruction are effective, keep them.
4. **Fix what's broken**: Address every issue in the feedback, especially misclassification patterns and weak relevance-boundary guidance.
5. **Add task-specific guidance**: Include concrete criteria specific to {query_type}/{doc_type} that help distinguish true positives from hard negatives.
6. **Be precise about boundaries**: Clearly define where positive ends and hard_negative begins for this task.
7. **Preserve the instruction style**: Keep the same style and structure as the current annotation instruction above (e.g. same level of detail, sectioning, and bullet conventions).

## Output Format
Output ONLY the rewritten annotation instruction (plain text). No JSON wrapping, no Markdown fences. This will be used as a system prompt for the annotator LLM."""


def refine_annotation_instruction(
    annotation_instruction: str,
    annotation_samples: List[Dict[str, Any]],
    feedback: str,
    task_name: str,
    llm_cfg: dict,
    logger=None,
    prompt_hook: Optional[Callable[[str], str]] = None,
    annotation_prompt_mode: str = ANNOTATION_PROMPT_MODE_MULTI_DOC_WITH_DOC_ID,
    include_annotation_samples: bool = True,
) -> str:
    """Use the LLM to rewrite the annotation instruction based on feedback.

    Parameters
    ----------
    prompt_hook : callable, optional
        If provided, called with the fully-built refine prompt string and
        expected to return the (possibly modified) prompt.  Used by
        ``HumanReviewer.review_refine_prompt`` to let a human edit the
        prompt before it is sent to the LLM.

    Returns the new instruction string.
    """
    log = logger or logging.getLogger(__name__)
    prompt = build_anno_instr_refine_prompt(
        annotation_instruction, annotation_samples, feedback, task_name,
        annotation_prompt_mode=annotation_prompt_mode,
        include_annotation_samples=include_annotation_samples,
    )
    if prompt_hook is not None:
        prompt = prompt_hook(prompt)

    refiner = Assistant(llm=llm_cfg)
    last_instruction = ""
    last_errors: list[str] = []
    max_refine_attempts = 2
    for attempt in range(1, max_refine_attempts + 1):
        raw = ""
        for response in refiner.run([{"role": "user", "content": prompt}]):
            raw = response[-1]["content"]

        instruction = raw.strip()
        last_instruction = instruction
        ok, errors = validate_instruction(
            instruction,
            annotation_prompt_mode=annotation_prompt_mode,
        )
        if ok:
            log.info(
                "Refined annotation instruction generated (length=%d chars, validation=ok, attempt=%d)",
                len(instruction), attempt,
            )
            return instruction

        last_errors = errors
        log.warning(
            "Refined annotation instruction failed validation on attempt %d/%d: %s",
            attempt, max_refine_attempts, "; ".join(errors),
        )

    raise ValueError(
        "Refined annotation instruction failed validation after "
        f"{max_refine_attempts} attempts: {'; '.join(last_errors)}. "
        f"Last output length={len(last_instruction)}"
    )


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    llm_cfg = {
        'model': 'gpt-4o-mini',
        'model_server': os.getenv('OPENAI_BASE_URL'),
        'api_key': os.getenv('OPENAI_API_KEY'),
    }

    # 1. Initialize the Annotation Instruction Generator Tool
    generator_tool = AnnotationInstructionGenerator()

    # 2. Generate the Evaluation Instruction for MSMarco (annotation logic: query + candidate documents -> score + positive/hard negative/easy negative)
    logging.info("=" * 80)
    logging.info("--- Generating Annotation Quality Assessment Instruction for arguana ---")
    logging.info("--- Annotation logic: input query + candidate documents -> LLM scores and classifies as positive/hard negative/easy negative ---")
    logging.info("=" * 80)
    tool_result = generator_tool.call(json5.dumps({'task_name': 'arguana'}), llm_cfg=llm_cfg)
    instruction = json5.loads(tool_result)['generated_instruction']
    logging.info(f"\nGenerated Evaluation Instruction:\n{instruction}\n")