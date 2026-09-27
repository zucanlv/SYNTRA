"""
Query Instruction Generator
============================
Generates a task-specific "Query Generation Style Guide" for a given IR task.
The guide is later embedded as a system prompt to steer an LLM toward producing
queries that align with the task's document corpus and gold-standard examples.

Caching
-------
Generated instructions are cached under ``Results/Instructions/`` using an
MD5 hash of the task definition tuple.  Identical task definitions re-use the
cached file; pass ``force_generate=True`` to bypass the cache.

Usage (standalone)
------------------
    python QueryInstr_Generator.py
"""

import hashlib
import json
import random
import json5
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

from Few_Shot_Formatter import format_examples, format_examples_queries_only, get_examples
from HighLevel_Def import HighLevel_Task_Definition

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs("Logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"Logs/QueryInstr_Generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)


# ---------------------------------------------------------------------------
# Query Extraction Process generator
# ---------------------------------------------------------------------------


def build_extraction_process_prompt(task_name: str) -> str:
    """Build a prompt that asks the LLM to analyse few-shot (doc, query) pairs
    and produce two **task-level** guideline sections:

    1. ``### Synthesizer Role`` — a 1-2 sentence role the LLM should adopt
       when generating queries (e.g. "You are an everyday internet user…").
    2. ``### Query Extraction Process`` — concrete steps for deriving a
       query from a new document in this task.

    Both sections must state general rules applicable to *any* document in the
    corpus, not observations specific to the example documents.
    """
    query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]
    few_shot_section = format_examples(get_examples(task_name))

    return f"""You are an expert in information retrieval and synthetic query synthesis.

## Task Context
- Query Type: {query_type}
- Document Type: {doc_type}
- Relevance Definition: {relevance_def}

## Few-Shot Examples — (Query ↔ Positive Document relevance pairs)
Each example shows a gold-standard {query_type} and a real {doc_type} that is relevant to it.
Use these pairs to infer how a real query author formulates the query-side expression that makes the document relevant.

{few_shot_section}

## Your Task — Two phases

**Phase 1 — Per-example analysis (think step-by-step for EACH example):**
For each (document, query) pair above, reason through:
1. Who is the real-world entity producing this type of query? What is their role, context, and goal?
2. What query-side purpose or expression would make the positive document relevant, and how is it formulated by the query author?
3. Which document details are authentic to include in this task's queries, and which details should be included, omitted, or transformed according to the query-document relationship demonstrated by the few-shot examples?
4. Does the query reflect the authentic voice, purpose, and framing of the query author, according to the query-document relationship demonstrated by the few-shot examples?

**Phase 2 — Cross-example generalisation (the actual output):**
After analysing all examples, output **exactly two sections** in the following order:

---

### Synthesizer Role

Write 1-2 sentences describing the role an LLM should *adopt* when synthesising {query_type} queries for this task.
The role must:
- Name who the producer is (e.g. "an everyday internet user", "a competitive debater", "a biomedical researcher")
- Describe what they are doing and what their goal is (e.g. "typing a query into a search engine to find an answer", "formulating a claim to argue in a debate round")
- Be grounded in the patterns observed across ALL few-shot examples — do NOT invent a role not supported by the data

---

### Query Extraction Process

Write a short, bulleted list of concrete steps the LLM should follow when reading a new {doc_type} and synthesising a {query_type} from it.
Each step must:
- Be stated as a general rule applicable to *any* {doc_type} in this corpus — do NOT reference specific entities, dates, or facts from the example documents
- Guide the LLM to infer a plausible query-side purpose or expression that would make the document relevant from the *producer's perspective* (as defined in Synthesizer Role)
- Specify which document-derived signals should shape the query, which details should be included/omitted/generalized, and how to frame the query according to the query-document relationship demonstrated by the few-shot examples

---

## Hard constraints for BOTH sections
- Every sentence must apply to any new {doc_type} encountered in this task, not just the examples above.
- Do NOT quote, paraphrase, or reference specific content from the example documents.
- The two sections together must be immediately usable as part of a system prompt without further editing.

## Output Format
Output ONLY the two sections above, in order, with their exact headings (`### Synthesizer Role` and `### Query Extraction Process`).
No JSON, no preamble, no per-example commentary, no additional text."""


def generate_extraction_process(task_name: str, llm_cfg: dict) -> str:
    """Call the LLM to produce the Synthesizer Role + Query Extraction Process sections.

    Returns the combined text of both sections (``### Synthesizer Role`` followed
    by ``### Query Extraction Process``), ready to prepend to the style guide.
    """
    prompt = build_extraction_process_prompt(task_name)
    agent = Assistant(llm=llm_cfg)
    raw = ""
    for response in agent.run([{"role": "user", "content": prompt}]):
        raw = response[-1]["content"]
    result = raw.strip()
    logging.info(
        "Generated Synthesizer Role + Query Extraction Process for task '%s' (len=%d chars)",
        task_name, len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@register_tool("query_instruction_generator")
class QueryInstructionGenerator(BaseTool):
    description = (
        "Generate a detailed system instruction for an LLM to generate queries "
        "from documents, based on a high-level task definition."
    )
    parameters = [
        {
            "name": "task_name",
            "type": "string",
            "description": (
                'The name of the task (e.g. "msmarco") from the high-level definitions.'
            ),
            "required": True,
        },
        {
            "name": "force_generate",
            "type": "boolean",
            "description": "Skip the instruction cache and always call the LLM.",
            "required": False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        params_dict = json5.loads(params)
        task_name: str = params_dict.get("task_name", "").lower()
        force_generate: bool = params_dict.get("force_generate", False)

        if task_name not in HighLevel_Task_Definition:
            return json5.dumps(
                {"error": f"Task '{task_name}' not found in HighLevel_Task_Definition."},
                ensure_ascii=False,
            )

        query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]

        # ------------------------------------------------------------------ #
        # Cache check                                                          #
        # ------------------------------------------------------------------ #
        def_content = f"{query_type}|{doc_type}|{relevance_def}"
        def_hash = hashlib.md5(def_content.encode("utf-8")).hexdigest()[:8]

        cache_dir = Path(__file__).parent / "Results" / "Instructions"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_files = sorted(
            cache_dir.glob(f"QueryInstr_{task_name}_{def_hash}_*.txt"),
            reverse=True,
        )

        if cache_files and not force_generate:
            cached_path = cache_files[0]
            logging.info(
                "Using cached query instruction for task '%s': %s",
                task_name,
                cached_path,
            )
            with open(cached_path, "r", encoding="utf-8") as f:
                instruction_output = f.read()
            return json5.dumps(
                {
                    "task_name": task_name,
                    "generated_instruction": instruction_output,
                    "cache_path": str(cached_path),
                    "from_cache": True,
                },
                ensure_ascii=False,
            )

        # ------------------------------------------------------------------ #
        # Build meta-prompt                                                    #
        # ------------------------------------------------------------------ #
        few_shot_section = format_examples_queries_only(get_examples(task_name))
        print("--------------------------------")
        print(few_shot_section)
        print("--------------------------------")

        meta_prompt = f"""### Role
You are a Senior Query Synthesist and Style Analyst. Your expertise lies in \
analyzing text patterns to create precise, actionable instructions for \
generating high-quality synthetic queries.

### Task Context:
- Target Output (Query Type): {query_type}
- Source Input (Document Type): {doc_type}
- Relevance Criteria: {relevance_def}

### Few-Shot Example Queries:
{few_shot_section}

### Instruction:
Based on the task context and the patterns observed in the few-shot examples, \
synthesize a "Task-Specific Style Guide". This guide will serve as the \
"gold standard" for an LLM to generate high-quality {query_type} queries for \
{doc_type} documents.

### Output Requirements:
You must actively analyze and distill features from both the Task Context \
and the Few-Shot Example Queries above. You mustobserve, abstract, and summarize what \
makes these queries fit the task, so the resulting guide can steer \
high-quality query generation. Address the following:

1. Linguistic DNA: From the examples, infer and state the tone (e.g. \
inquisitive, keyword-heavy), structural syntax, typical length, and any \
other recurring linguistic traits.
2. Relevance Heuristics: From the task context and examples, identify \
which information patterns in a {query_type} should be emphasised to \
satisfy the relevance criteria ({relevance_def}).
3. Negative Constraints: List "anti-patterns"—phrasing, styles, or \
excessive detail that would harm data purity and should be avoided.


### Output Format:
Produce a compact, bulleted instruction set suitable for embedding in a \
system prompt for synthetic query generation. Organise your output under \
the three subheadings above: Linguistic DNA, Relevance Heuristics, and \
Negative Constraints (with bullet points listed under each). 
Output only these subheadings and bullet points in plain text; no other text.
"""

        # ------------------------------------------------------------------ #
        # LLM call                                                             #
        # ------------------------------------------------------------------ #
        # Priority: kwargs["llm_cfg"] (call-time) > self.cfg (init-time) > env vars
        _kwarg_cfg: dict = kwargs.get("llm_cfg", {})
        llm_cfg = {
            "model": (
                _kwarg_cfg.get("model")
                or "gpt-4o-mini"
            ),
            "model_server": (
                _kwarg_cfg.get("model_server")
                or os.getenv("OPENAI_BASE_URL")
            ),
            "api_key": (
                _kwarg_cfg.get("api_key")
                or os.getenv("OPENAI_API_KEY")
            ),
        }

        # Step 1: Generate task-level Query Extraction Process via CoT analysis
        # of few-shot (doc, query) pairs.  This is a separate LLM call so the
        # extraction reasoning has a focused objective.
        logging.info("Generating Query Extraction Process for task '%s' ...", task_name)
        extraction_process = generate_extraction_process(task_name, llm_cfg)

        # Step 2: Generate the existing style guide (Linguistic DNA, Relevance
        # Heuristics, Negative Constraints) from query-only few-shot examples.
        designer = Assistant(llm=llm_cfg)
        style_guide = ""
        for response in designer.run([{"role": "user", "content": meta_prompt}]):
            style_guide = response[-1]["content"]

        # Combine: Extraction Process is placed first so the LLM reads "how to
        # think about the document" before "how to format the query".
        instruction_output = extraction_process.strip() + "\n\n" + style_guide.strip()

        # ------------------------------------------------------------------ #
        # Persist to cache                                                     #
        # ------------------------------------------------------------------ #
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = cache_dir / f"QueryInstr_{task_name}_{def_hash}_{timestamp}.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(instruction_output)
        logging.info("Generated query instruction saved to: %s", save_path)

        return json5.dumps(
            {
                "task_name": task_name,
                "generated_instruction": instruction_output,
                "cache_path": str(save_path),
                "from_cache": False,
            },
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Self-refine: evaluate & refine query instructions
# ---------------------------------------------------------------------------


def _format_query_doc_pairs_for_prompt(query_doc_pairs: list, max_pairs: int = 5) -> str:
    """Render a sample of (positive document, generated query) pairs.

    Each item in *query_doc_pairs* must be a dict with at least ``"query"``
    and ``"doc"`` keys.
    """
    sample = random.sample(query_doc_pairs, min(max_pairs, len(query_doc_pairs)))
    lines = []
    for i, pair in enumerate(sample, 1):
        positive_doc = (pair.get("doc") or "").strip()
        query_text = (pair.get("query") or "").strip()
        lines.append(
            f"Pair {i}:\n"
            f"  Positive document: {positive_doc}\n"
            f"  Generated query: {query_text}"
        )
    return "\n\n".join(lines)


def build_query_instr_evaluate_prompt(
    query_instruction: str,
    query_doc_pairs: list,
    task_name: str,
) -> str:
    """Build a prompt asking the LLM to evaluate generated queries against
    the task's few-shot gold standard and task context.

    Parameters
    ----------
    query_doc_pairs : list of dict
        Each item must have ``"query"`` (str) and ``"doc"`` (str) keys.
        The source document is used to evaluate whether the generated query
        is appropriately abstracted from document-specific details.
    """
    query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]
    examples = get_examples(task_name)
    few_shot_queries_section = format_examples_queries_only(examples)

    pairs_section = _format_query_doc_pairs_for_prompt(query_doc_pairs, max_pairs=5)

    return f"""You are a senior quality evaluator for an information retrieval data synthesis pipeline.

## Your Task
Evaluate whether the **generated queries** below match the expected style, tone, and characteristics of real **{query_type}** queries for this task.

## Step-by-step guideline:
1. **Understand the task.** Be clear on what good {query_type} queries look like in this setting and what distinguishes them.
2. **Evaluate the (document, query) pairs one by one.** For each pair, evaluate the generated query against ALL five criteria below and record issues as you go.
    i. **Compare to the reference set.** Does the query match the golden queries in style, tone, length, and the level of abstraction/specificity?
    ii. **Apply the criteria.** Record any issue clearly per pair (e.g., "Pair 3: over-specifies document details, too long").
3. **Summarize.** Provide a concise summary of the main issues — or confirm quality is acceptable. If citing a problematic pair, use it only as evidence; the feedback itself must generalize the underlying task-level issue rather than preserve sample-specific entities, dates, facts, or wording. This summary will be used to refine a reusable query synthesis instruction.
4. **Output.** Respond strictly in the required JSON output format.

## Task Context
- Query Type: {query_type}
- Document Type: {doc_type}
- Relevance Definition: {relevance_def}

## Few-Shot Golden Queries (style + abstraction benchmark — compare generated queries to these)
{few_shot_queries_section}

## Current Query Synthesis Instruction (used to generate the queries below)
{query_instruction}

## Generated (Positive Document ↔ Query) Pairs (sample)
{pairs_section}

## Evaluation Criteria (check each rigorously)
1. **Style Match**: Do the generated queries resemble the few-shot example queries in tone, length, vocabulary, and structure?
2. **Task Alignment**: Are the queries the kind of {query_type} that would realistically be issued against {doc_type} documents?
3. **Naturalness**: Do the queries read like something a real producer of this query type would write, or do they sound artificial/templated?
4. **User Authenticity**: Does each query feel like it was genuinely produced by the role described in the Synthesizer Role section of the instruction — i.e., someone acting in that role *without* having access to the source document? The key question is whether they are *authentic to the producer's perspective and intent in this task*. Penalise queries that merely describe or echo document content rather than reflecting genuine producer intent.
5. **Instruction Effectiveness**: Does the current instruction successfully guide the LLM on all four criteria above, including the Synthesizer Role and Query Extraction Process sections?

## Scoring Rubric (assign exactly one integer score 0–3)
> **Important**: Base your score on the holistic pattern across **all five criteria** above.

- **Score 3 — Production-Ready**: Generated queries closely match the golden queries in style, authenticity, naturalness, and the query-document relationship demonstrated by the examples. The Synthesizer Role and Query Extraction Process sections guide the LLM to produce queries from the genuine producer's perspective.
- **Score 2 — Acceptable**: Queries are broadly on-style and authentic, with only minor isolated deviations from the golden examples' producer perspective or query-document relationship. Only small targeted adjustments are needed.
- **Score 1 — Needs Revision**: Systematic issues across most queries — e.g., consistently using the wrong producer role, wrong level of specificity, wrong register, wrong relationship to the positive document, or otherwise missing the authentic producer perspective demonstrated in the golden examples. The Synthesizer Role or Query Extraction Process section is missing, vague, or not guiding correctly.
- **Score 0 — Unusable**: Queries are fundamentally off-task, incoherent, or incompatible with the task's expected query type and query-document relationship.

## Output Format (MANDATORY — valid JSON only, no Markdown)
Output a single JSON object:
{{
  "overall_score": 0-3,
  "passed": true if overall_score >= 2 else false,
  "feedback": "Concise summary of the main issues. Be specific: cite example pairs that are problematic and explain WHY.",
  "style_issues": "Specific style mismatches vs few-shot examples, or 'none'.",
  "authenticity_issues": "Whether queries fail to reflect the genuine producer perspective (e.g. just echo document content, wrong role, wrong intent), or 'none'.",
  "instruction_issues": "What the query instruction gets wrong or misses, or 'none'.",
  "suggestions": "Concrete, actionable suggestions for improving the query instruction."
}}

IMPORTANT: A "passed" verdict (overall_score >= 2) means the queries are production-ready. If there is ANY systematic authenticity or style mismatch across multiple pairs, assign score <= 1 and set passed=false."""


def evaluate_query_instruction(
    query_instruction: str,
    query_doc_pairs: list,
    task_name: str,
    llm_cfg: dict,
    logger=None,
) -> tuple:
    """Evaluate generated queries against task context and few-shot style.

    Parameters
    ----------
    query_doc_pairs : list of dict
        Each item must have ``"query"`` (str) and ``"doc"`` (str) keys.

    Returns ``(passed: bool, feedback: str, eval_details: dict)``.
    """
    log = logger or logging.getLogger(__name__)
    prompt = build_query_instr_evaluate_prompt(query_instruction, query_doc_pairs, task_name)

    evaluator = Assistant(llm=llm_cfg)
    raw = ""
    for response in evaluator.run([{"role": "user", "content": prompt}]):
        raw = response[-1]["content"]

    eval_details: dict = {}
    try:
        parsed = json5.loads(raw)
        score = parsed.get("overall_score", 0)
        passed = score >= 2
        parts = []
        if parsed.get("feedback"):
            parts.append(parsed["feedback"])
        if parsed.get("style_issues") and parsed["style_issues"].lower() != "none":
            parts.append(f"Style issues: {parsed['style_issues']}")
        if parsed.get("authenticity_issues") and parsed["authenticity_issues"].lower() != "none":
            parts.append(f"Authenticity issues: {parsed['authenticity_issues']}")
        if parsed.get("instruction_issues") and parsed["instruction_issues"].lower() != "none":
            parts.append(f"Instruction issues: {parsed['instruction_issues']}")
        if parsed.get("suggestions"):
            parts.append(f"Suggestions: {parsed['suggestions']}")
        feedback = "\n".join(parts)
        eval_details = {
            "score": score,
            "feedback": parsed.get("feedback", ""),
            "style_issues": parsed.get("style_issues", ""),
            "authenticity_issues": parsed.get("authenticity_issues", ""),
            "instruction_issues": parsed.get("instruction_issues", ""),
            "suggestions": parsed.get("suggestions", ""),
        }
        log.info("Query instruction evaluation: passed=%s score=%s", passed, score)
    except Exception as e:
        log.warning("Failed to parse evaluation JSON: %s — treating as not passed", e)
        passed = False
        feedback = f"Evaluation parse error. Raw output:\n{raw[:1000]}"

    return passed, feedback, eval_details


def build_query_instr_refine_prompt(
    query_instruction: str,
    query_doc_pairs: list,
    feedback: str,
    task_name: str,
) -> str:
    """Build a prompt asking the LLM to rewrite the query instruction.

    Parameters
    ----------
    query_doc_pairs : list of dict
        Each item must have ``"query"`` (str) and ``"doc"`` (str) keys.
    """
    query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]
    few_shot_queries_section = format_examples_queries_only(get_examples(task_name))

    pairs_section = _format_query_doc_pairs_for_prompt(query_doc_pairs, max_pairs=8)

    return f"""You are a prompt engineering expert specializing in information retrieval query synthesis.

## Context
We have a "Query Synthesis Instruction" that is embedded as a system prompt to steer an LLM toward producing **{query_type}** queries for **{doc_type}** documents.

- Relevance Definition: {relevance_def}

## Few-Shot Golden Queries (the gold standard — queries MUST match this style and abstraction level)
{few_shot_queries_section}

## Problem
The current instruction was evaluated and received the following feedback:

### Evaluation Feedback
{feedback}

### Sample (Positive Document ↔ Generated Query) Pairs (produced by the current instruction)
{pairs_section}

### Current Query Synthesis Instruction
{query_instruction}

## Your Task
Rewrite the Query Synthesis Instruction to fix all issues identified in the feedback. The rewritten instruction must produce queries that closely match the few-shot examples in style, tone, length, and domain conventions.

## Requirements
1. **Preserve what works**: If parts of the current instruction are effective, keep them.
2. **Fix what's broken**: Address every issue in the feedback — including authenticity mismatches where generated queries fail to reflect the genuine producer perspective (e.g., they just echo document content, adopt the wrong register, or lack the intent of the role described).
3. **Be specific**: Instead of "use appropriate tone", specify exactly what tone (e.g., "**Tone:** Analytical, assertive, and occasionally contentious to reflect a strong stance.").
4. **Update the Synthesizer Role section if needed**: If the feedback indicates the LLM is not adopting the right producer identity, sharpen the role description to better convey who is writing the query and why.
5. **Update the Query Extraction Process section if needed**: If the feedback indicates queries are not reflecting the authentic producer perspective, revise the extraction steps to more explicitly guide the LLM to think from the producer's viewpoint.
6. **Format**: Output as a compact, bulleted query style guidelines set directly embeddable as a system prompt. No JSON wrapping. Keep the same style and structure as the current query style guidelines above (e.g. same level of detail, sectioning, and bullet conventions).

## Output Format
Output ONLY the rewritten instruction text (plain text with bullet points and section headings). No JSON, no Markdown fences, no commentary."""


def refine_query_instruction(
    query_instruction: str,
    query_doc_pairs: list,
    feedback: str,
    task_name: str,
    llm_cfg: dict,
    logger=None,
    prompt_hook: Optional[Callable[[str], str]] = None,
) -> str:
    """Use the LLM to rewrite the query instruction based on feedback.

    Parameters
    ----------
    query_doc_pairs : list of dict
        Each item must have ``"query"`` (str) and ``"doc"`` (str) keys.
    prompt_hook : callable, optional
        If provided, called with the fully-built refine prompt string and
        expected to return the (possibly modified) prompt.  Used by
        ``HumanReviewer.review_refine_prompt`` to let a human edit the
        prompt before it is sent to the LLM.

    Returns the new instruction string.
    """
    log = logger or logging.getLogger(__name__)
    prompt = build_query_instr_refine_prompt(
        query_instruction, query_doc_pairs, feedback, task_name,
    )
    if prompt_hook is not None:
        prompt = prompt_hook(prompt)

    refiner = Assistant(llm=llm_cfg)
    raw = ""
    for response in refiner.run([{"role": "user", "content": prompt}]):
        raw = response[-1]["content"]

    instruction = raw.strip()
    log.info("Refined query instruction generated (length=%d chars)", len(instruction))
    return instruction


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    llm_cfg = {
        "model": "gemini-3-flash-preview-thinking",
        "model_server": os.getenv("OPENAI_BASE_URL"),
        "api_key": os.getenv("OPENAI_API_KEY"),
    }

    generator_tool = QueryInstructionGenerator()

    task_name = "arguana"
    logging.info(f"--- Generating Query Instruction for {task_name} ---")
    tool_result = generator_tool.call(
        json5.dumps({"task_name": task_name}),
        llm_cfg=llm_cfg,
    )
    result = json5.loads(tool_result)
    instruction = result["generated_instruction"]
    from_cache = result.get("from_cache", False)
    logging.info("From cache: %s", from_cache)
    logging.info("Generated Instruction:\n%s\n", instruction)
