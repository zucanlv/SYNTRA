"""
High-Level Task Definition Generator
=====================================
Infers a high-level IR task definition — (query_type, doc_type, relevance_definition) —
from a corpus's few-shot (query, Positive, Hard Negative) examples by prompting an LLM
with msmarco and arguana as gold-standard two-shot demonstrations.

The generated (query_type, doc_type, relevance_definition) tuple can be directly inserted
into HighLevel_Def.py to bootstrap new corpora without hand-crafting task definitions.

Design
------
1. **Two-shot in-context demonstrations**: msmarco and arguana examples are shown alongside
   their known HighLevel_Task_Definition, grounding the LLM on the expected output quality
   and format.

2. **Chain-of-thought extraction**: The LLM is guided through a three-step analysis —
   (a) characterise the query style/intent, (b) characterise the document type/domain,
   (c) use the contrast between Positive and Hard Negative examples to sharpen the
   relevance definition — before producing the final JSON.

3. **Self-refine loop**: After generation, an evaluator LLM checks whether the produced
   definition correctly explains every example pair in the corpus.  If not, a refiner
   rewrites the definition based on specific failure feedback (up to ``max_refine_attempts``
   rounds).

4. **Caching**: Results are stored under ``Results/Instructions/`` using an MD5 hash of
   the serialised examples so identical inputs skip the LLM call.

Caching
-------
Cache files are named ``HighLevel_{task_name}_{hash}_{timestamp}.json``.  The most recent
matching file is returned unless ``force_generate=True`` is passed.

Usage (standalone)
------------------
    python HighLevel_Generator.py --task trec-covid
    python HighLevel_Generator.py --task trec-covid --force
    python HighLevel_Generator.py --task trec-covid --model gpt-4o --max-refine 3

    python HighLevel_Generator.py \
    --task fiqa \
    --model deepseek-v4-pro \
    --model-server https://api.deepseek.com \
    --api-key "$DPSK_API_KEY" \
    --max-refine 5 \
    --demo-max-examples 5 \
    --force
"""

import argparse
import hashlib
import json
import json5
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qwen_agent.agents import Assistant

from Few_Shot_Example import Few_Shot_Example
from Few_Shot_Formatter import format_examples, get_examples
from HighLevel_Def import HighLevel_Task_Definition
from SelfRefine import RefineAttempt, SelfRefineLoop

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HighLevelConfig:
    """Configuration for high-level task definition generation."""

    task_name: str = ""

    # Output
    output_dir: str = "Results/Instructions"
    log_dir: str = "Logs"

    # LLM
    model: str = "gpt-4o-mini"
    model: str = "gemini-3-flash-preview-thinking"
    model_server: Optional[str] = None
    api_key: Optional[str] = None

    # Self-refine
    max_refine_attempts: int = 3

    # Prompt controls
    demo_max_examples: int = 5

    # Cache bypass
    force_generate: bool = False

    # User feedback: when True, after evaluate fails the user can edit feedback
    # and optionally supply a custom refine prompt via input()
    enable_user_feedback: bool = False

    def get_llm_cfg(self) -> dict:
        return {
            "model": self.model,
            "model_server": self.model_server or os.getenv("OPENAI_BASE_URL", ""),
            "api_key": self.api_key or os.getenv("OPENAI_API_KEY", ""),
        }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _format_demo_block(task_name: str, max_examples: int = 3) -> str:
    """Render a single two-shot demonstration block for *task_name*.

    The block shows the few-shot examples paired with the known HighLevel_Task_Definition,
    so the LLM can learn the mapping from examples to abstract definition.
    """
    examples = get_examples(task_name)
    query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]
    examples_text = format_examples(examples, max_examples=max_examples)

    return f"""### Demonstration — corpus: "{task_name}"

**Few-Shot Examples:**
{examples_text}

**Extracted High-Level Task Definition:**
- query_type: "{query_type}"
- doc_type: "{doc_type}"
- relevance_definition: "{relevance_def}"
"""


def build_extract_prompt(
    task_name: str,
    examples: List[Dict],
    demo_max_examples: int = 5,
) -> str:
    """Build the main extraction prompt with two-shot demonstrations and CoT instructions.

    The prompt guides the LLM through a structured three-step analysis:
    1. Query Analysis — style, length, intent, domain
    2. Document Analysis — type, structure, domain of Positive docs
    3. Contrastive Analysis — what distinguishes Positive from Hard Negative

    These steps are designed to anchor the output to the actual signal in the examples
    rather than generic IR conventions.

    Parameters
    ----------
    task_name : str
        Name of the target corpus (used for logging only; no pre-existing definition needed).
    examples : list of dict
        Raw few-shot examples for the target corpus (query, Positive, Hard Negative).
    """
    # Build two-shot demonstrations from corpora that have known definitions
    demo_corpora = [k for k in ("msmarco", "arguana") if k in HighLevel_Task_Definition and k != task_name]
    demo_blocks = "\n\n".join(
        _format_demo_block(c, max_examples=demo_max_examples)
        for c in demo_corpora
    )

    target_examples_text = format_examples(examples)

    has_hard_negatives = any(ex.get("Hard Negative", "").strip() for ex in examples)
    contrastive_instruction = (
        "- **Contrastive Analysis**: Compare each Positive document against its Hard Negative. "
        "Hard Negatives are topically related but NOT relevant — identify the precise criterion "
        "that distinguishes a relevant document from a hard-but-wrong one. This is the single "
        "most important signal for writing a precise relevance_definition."
        if has_hard_negatives
        else "- **Contrastive Analysis**: No Hard Negative examples are provided. Focus on "
        "what specific aspect of the Positive documents makes them the ideal match for the queries."
    )

    return f"""You are an expert Information Retrieval task analyst. Your job is to precisely characterise an IR task from its labeled examples.

## Two-Shot Demonstrations
Study these examples carefully — each shows how few-shot examples map to an abstract high-level task definition:

{demo_blocks}

---

## Your Task: Extract the High-Level Task Definition for the target corpus

### Few-Shot Examples to Analyse
{target_examples_text}

---

### Step-by-Step Analysis Instructions

Work through the following three steps in your reasoning. Be specific and cite evidence from the examples above.

**Step 1 — Query Analysis**
- What is the average length of the queries (short/medium/long)?
- What is the vocabulary style (natural language, keyword-style, formal, argumentative, domain-specific)?
- What is the user intent (fact-seeking, opinion-seeking, argument-seeking, research-oriented, ...)?
- What is the domain (general web, biomedical, legal, debate, ...)?
- Summarise into a precise 2-5 word phrase for `query_type`.

**Step 2 — Document Analysis**
- What type/structure do the Positive documents have (news articles, scientific abstracts, debate arguments, forum answers, ...)?
- What domain/register are they in?
- How long are they on average?
- Summarise into a precise 2-5 word phrase for `doc_type`.

{contrastive_instruction}

**Step 3 — Relevance Definition Synthesis**
Combine the above analysis into ONE precise sentence following this exact template:
"Given a query (<query_type>) and a document (<doc_type>), the document is relevant if <specific criterion based on the contrastive analysis>."

The sentence must:
- Fill in <query_type> and <doc_type> from Steps 1 & 2.
- State the exact criterion (not just "it answers the query") derived from the contrastive analysis.
- Be specific enough that a language model reading only this sentence would produce queries and documents matching the style in the examples above.

---

### Output Format (MANDATORY — valid JSON only, no Markdown fences)
After your step-by-step reasoning, output a single JSON object:
{{
  "query_type": "2-5 word phrase",
  "doc_type": "2-5 word phrase",
  "relevance_definition": "Given a query (...) and a document (...), the document is relevant if ..."
}}

Output the JSON as the LAST thing in your response. Do not include any text after the closing brace."""


def build_evaluate_prompt(
    task_name: str,
    examples: List[Dict],
    task_def: Dict,
) -> str:
    """Build a prompt that evaluates whether the generated definition fits the examples.

    The evaluator checks each example pair against the definition to identify gaps,
    over-generalisations, or mischaracterisations.

    Parameters
    ----------
    task_def : dict
        The generated definition with keys: query_type, doc_type, relevance_definition.
    """
    examples_text = format_examples(examples)
    query_type = task_def.get("query_type", "")
    doc_type = task_def.get("doc_type", "")
    relevance_def = task_def.get("relevance_definition", "")

    return f"""You are a senior quality evaluator for an information retrieval data synthesis pipeline.

## Your Task
Evaluate whether the high-level task definition below correctly and precisely characterises the IR task evidenced by the few-shot examples for the target corpus.

## Few-Shot Examples
{examples_text}

## Proposed High-Level Task Definition
- query_type: "{query_type}"
- doc_type: "{doc_type}"
- relevance_definition: "{relevance_def}"

## Evaluation Criteria (check each rigorously)

1. **Query Type Accuracy**: Does `query_type` precisely describe the style, intent, and domain of the queries in the examples? It should not be too generic (e.g., "query") nor too narrow.

2. **Document Type Accuracy**: Does `doc_type` precisely describe the Positive documents' type, structure, and domain? It should distinguish these documents from documents in other IR tasks.

3. **Relevance Definition Precision**: Does `relevance_definition` correctly explain why each Positive document is relevant to its query? Test it mentally: if you applied this definition as a filter, would it accept all Positive documents and reject all Hard Negatives?

4. **Task Distinctiveness**: Could this definition be confused with another well-known IR task? If so, the definition is too vague.

5. **Actionability**: Is the definition specific enough that an LLM reading only this sentence would know what kind of queries to generate and what kind of documents to retrieve?

## Step-by-step Evaluation
Go through each example one by one. For each:
- Does the query fit `query_type`?
- Does the Positive document fit `doc_type` and satisfy `relevance_definition`?
- (If Hard Negative present) Does the Hard Negative correctly fail `relevance_definition`?

Record any per-example issues, then summarise.

## Output Format (MANDATORY — valid JSON only, no Markdown)
{{
  "passed": true | false,
  "overall_score": 1-4,
  "feedback": "Concise summary of the main issues, citing specific examples. Or confirm quality if none.",
  "query_type_issues": "Specific issues with query_type, or 'none'.",
  "doc_type_issues": "Specific issues with doc_type, or 'none'.",
  "relevance_def_issues": "Specific issues with relevance_definition (false accepts, false rejects, vagueness), or 'none'.",
  "suggestions": "Concrete, actionable rewrites or additions to fix the identified issues."
}}

IMPORTANT: A "passed" verdict (overall_score >= 3) means the definition is production-ready. Any systematic mismatch with the examples must set passed=false."""


def build_refine_prompt(
    task_name: str,
    examples: List[Dict],
    task_def: Dict,
    feedback: str,
) -> str:
    """Build a prompt that rewrites the task definition based on evaluation feedback.

    Parameters
    ----------
    task_def : dict
        The current (flawed) definition.
    feedback : str
        Consolidated feedback from the evaluator.
    """
    examples_text = format_examples(examples)
    query_type = task_def.get("query_type", "")
    doc_type = task_def.get("doc_type", "")
    relevance_def = task_def.get("relevance_definition", "")

    # Include demonstration definitions as quality anchors
    demo_corpora = [k for k in ("msmarco", "arguana") if k in HighLevel_Task_Definition and k != task_name]
    anchor_lines = []
    for i, c in enumerate(demo_corpora, start=1):
        qt, dt, rd = HighLevel_Task_Definition[c]
        anchor_lines.append(
            f'- Reference definition {i}: query_type="{qt}", doc_type="{dt}"\n  relevance_definition="{rd}"'
        )
    anchor_block = "\n".join(anchor_lines)

    return f"""You are a prompt engineering expert specialising in information retrieval task characterisation.

## Context
We have a high-level task definition for the target corpus that was evaluated and found to have issues.

## Few-Shot Examples (ground truth)
{examples_text}

## Current (Flawed) Definition
- query_type: "{query_type}"
- doc_type: "{doc_type}"
- relevance_definition: "{relevance_def}"

## Evaluation Feedback
{feedback}

## Quality Anchors (reference definitions for other corpora)
{anchor_block}

## Your Task
Rewrite the high-level task definition for the target corpus to fix ALL issues identified in the feedback.

## Requirements
1. **Fix every issue**: Address each problem explicitly mentioned in the feedback.
2. **Stay grounded**: Every claim in the definition must be directly supported by at least one example.
3. **Maintain specificity**: query_type and doc_type must be 2-5 word phrases. relevance_definition must be one precise sentence.
4. **Follow the template**: "Given a query (<query_type>) and a document (<doc_type>), the document is relevant if <specific criterion>."

## Output Format (MANDATORY — valid JSON only, no Markdown fences)
{{
  "query_type": "2-5 word phrase",
  "doc_type": "2-5 word phrase",
  "relevance_definition": "Given a query (...) and a document (...), the document is relevant if ..."
}}"""


# ---------------------------------------------------------------------------
# User feedback (interactive input)
# ---------------------------------------------------------------------------


def show_evaluation_result(
    feedback: str,
    score: int,
    passed: bool,
    task_name: str,
    task_def: Optional[Dict] = None,
) -> None:
    """Print evaluation result (feedback, score, passed) and current task definition to the terminal."""
    print("\n" + "=" * 60)
    print(f"[Evaluation Result] Task \"{task_name}\"")
    print("=" * 60)
    if task_def is not None:
        print("  Current definition (this round):")
        print(f"    query_type         : {task_def.get('query_type', '')}")
        print(f"    doc_type           : {task_def.get('doc_type', '')}")
        print(f"    relevance_definition: {task_def.get('relevance_definition', '')}")
        print("  ---")
    print(f"  passed : {passed}")
    print(f"  score  : {score}")
    print("  feedback:")
    print(feedback)
    print("-" * 60)


def ask_user_pass_definition(task_name: str, logger: Optional[logging.Logger] = None) -> bool:
    """Ask the user whether to accept the current High-Level Task Definition as passed.

    Called after showing evaluation result when ``enable_user_feedback`` is True.
    The actual pass/fail is decided by the user, not by the LLM evaluator.

    Returns
    -------
    True if the user accepts (pass), False otherwise.
    """
    log = logger or logging.getLogger(__name__)
    while True:
        ans = input(f'Pass this High-Level Task Definition for "{task_name}"? (y/n): ').strip().lower()
        if ans in ("y", "yes"):
            log.info("User accepted the definition as passed.")
            return True
        if ans in ("n", "no"):
            log.info("User did not pass the definition; will refine.")
            return False
        print("Please enter y or n.")


def get_user_feedback_interactive(
    original_feedback: str,
    score: int,
    passed: bool,
    task_name: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, Optional[str]]:
    """Ask the user whether to modify feedback and/or refine prompt via y/n input().

    After each evaluate, the caller should show feedback/score/passed (e.g. via
    show_evaluation_result). Then call this only when about to refine (not passed
    and attempt < max). Returns the feedback and optional refine prompt to use.

    Parameters
    ----------
    original_feedback : str
        The feedback string from the evaluator LLM.
    score : int
        overall_score from the evaluator.
    passed : bool
        Whether the evaluation passed.
    task_name : str
        Task name (for display).
    logger : Logger, optional

    Returns
    -------
    (feedback_to_use, refine_prompt_override)
        feedback_to_use: the feedback string to pass to refine (user-edited or original).
        refine_prompt_override: custom full prompt for the refiner LLM, or None to use the default.
    """
    log = logger or logging.getLogger(__name__)
    new_feedback = original_feedback
    modify_fb = input("Modify feedback? (y/n): ").strip().lower()
    if modify_fb == "y":
        new_feedback = input("Enter new feedback: ").strip() or original_feedback
        log.info("User replaced evaluation feedback with custom text.")
    modify_prompt = input("Modify refine prompt? (y/n): ").strip().lower()
    custom_prompt = None
    if modify_prompt == "y":
        custom_prompt = input("Enter custom refine prompt: ").strip() or None
        if custom_prompt:
            log.info("User supplied a custom refine prompt.")
    return new_feedback, custom_prompt


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------


def _run_llm(prompt: str, llm_cfg: dict) -> str:
    """Call the LLM and return the raw text response."""
    agent = Assistant(llm=llm_cfg)
    raw = ""
    for response in agent.run([{"role": "user", "content": prompt}]):
        raw = response[-1]["content"]
    return raw


def _extract_json_from_response(raw: str, logger: logging.Logger) -> Optional[Dict]:
    """Extract the last JSON object from a potentially verbose LLM response.

    The extraction prompt encourages chain-of-thought reasoning followed by a
    terminal JSON block.  We search from the end to find the last ``{`` that
    opens a parseable JSON object.
    """
    # Try to find the last JSON object in the response
    last_brace = raw.rfind("}")
    if last_brace == -1:
        logger.warning("No closing brace found in LLM response.")
        return None

    # Walk backwards from the last closing brace to find its matching opening brace
    depth = 0
    start = -1
    for i in range(last_brace, -1, -1):
        if raw[i] == "}":
            depth += 1
        elif raw[i] == "{":
            depth -= 1
            if depth == 0:
                start = i
                break

    if start == -1:
        logger.warning("Could not find matching opening brace in LLM response.")
        return None

    candidate = raw[start : last_brace + 1]
    try:
        return json5.loads(candidate)
    except Exception as e:
        logger.warning("Failed to parse extracted JSON candidate: %s\nCandidate: %s", e, candidate[:500])
        return None


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------


def generate_highlevel_task(
    task_name: str,
    examples: List[Dict],
    llm_cfg: dict,
    max_refine_attempts: int = 3,
    demo_max_examples: int = 3,
    enable_user_feedback: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[Dict], bool]:
    """Generate a high-level task definition from few-shot examples.

    Uses a self-refine loop: extract → evaluate → [optional user edit] → refine (up to *max_refine_attempts*).

    Parameters
    ----------
    task_name : str
        Name of the corpus / IR task.
    examples : list of dict
        Few-shot examples with keys: query, Positive, (optionally) Hard Negative.
    llm_cfg : dict
        LLM configuration dict for qwen_agent.
    max_refine_attempts : int
        Total attempts including the initial generation.
    demo_max_examples : int
        Maximum number of examples shown in each reference demonstration block.
    enable_user_feedback : bool
        If True, after each failed evaluation the user can edit feedback and optionally
        supply a custom refine prompt via input().
    logger : Logger, optional

    Returns
    -------
    (task_def, passed) where task_def is a dict with query_type, doc_type,
    relevance_definition, and passed is a bool.
    """
    log = logger or logging.getLogger(__name__)

    # ---- define callables for SelfRefineLoop --------------------------------

    def generate_fn(artifact: Optional[Dict]) -> Dict:
        """artifact is the previous (rejected) definition or None on first call."""
        if artifact is not None:
            log.info("Using refined high-level task definition from previous attempt.")
            return artifact

        # First attempt: extract from scratch using CoT prompt
        prompt = build_extract_prompt(task_name, examples, demo_max_examples=demo_max_examples)
        log.info("Calling LLM to extract high-level task definition …")

        raw = _run_llm(prompt, llm_cfg)
        parsed = _extract_json_from_response(raw, log)
        if parsed is None:
            log.warning("LLM did not return parseable JSON; returning empty definition.")
            parsed = {"query_type": "", "doc_type": "", "relevance_definition": ""}
        return parsed

    def evaluate_fn(task_def: Dict) -> Tuple[bool, str, int]:
        """Returns (passed, feedback, overall_score)."""
        prompt = build_evaluate_prompt(task_name, examples, task_def)
        log.info("Evaluating generated definition …")
        raw = _run_llm(prompt, llm_cfg)
        parsed = _extract_json_from_response(raw, log)
        if parsed is None:
            log.warning("Evaluator returned unparseable JSON — treating as not passed.")
            return False, f"Evaluation parse error. Raw output:\n{raw[:800]}", 0

        passed = bool(parsed.get("passed", False))
        score = int(parsed.get("overall_score", 0))
        parts = []
        for key in ("feedback", "query_type_issues", "doc_type_issues", "relevance_def_issues", "suggestions"):
            val = parsed.get(key, "")
            if val and str(val).lower() not in ("none", ""):
                parts.append(f"[{key}] {val}")
        feedback = "\n".join(parts) or "No feedback provided."
        log.info("Evaluation result: passed=%s, score=%s", passed, score)
        return passed, feedback, score

    def refine_fn(_artifact: Any, outputs: Dict, feedback: str, refine_prompt_override: Optional[str] = None) -> Dict:
        """outputs is the task_def from generate_fn; artifact is unused (was None)."""
        if refine_prompt_override:
            prompt = refine_prompt_override
        else:
            prompt = build_refine_prompt(task_name, examples, outputs, feedback)
        log.info("Refining definition based on feedback …")
        raw = _run_llm(prompt, llm_cfg)
        refined = _extract_json_from_response(raw, log)
        if refined is None:
            log.warning("Refiner returned unparseable JSON — keeping previous definition.")
            return outputs
        return refined

    # ---- run loop (with or without user feedback) ---------------------------

    if enable_user_feedback:
        # Custom loop: evaluate → show result → user decides pass (y/n); if not pass → modify feedback/prompt → refine
        artifact = None
        history: List[RefineAttempt] = []
        for attempt_num in range(1, max_refine_attempts + 1):
            if artifact is None:
                log.info("[SelfRefine] Attempt %d/%d — generating outputs ...", attempt_num, max_refine_attempts)
                outputs = generate_fn(artifact)
            else:
                outputs = artifact  # use refined definition from previous round
            log.info("[SelfRefine] Attempt %d/%d — evaluating outputs ...", attempt_num, max_refine_attempts)
            llm_passed, feedback, score = evaluate_fn(outputs)
            history.append(RefineAttempt(attempt=attempt_num, artifact=artifact, outputs=outputs, passed=llm_passed, feedback=feedback))
            show_evaluation_result(feedback, score, llm_passed, task_name, task_def=outputs)
            user_passed = ask_user_pass_definition(task_name, log)
            if user_passed:
                log.info("[SelfRefine] Attempt %d/%d — user accepted, PASSED.", attempt_num, max_refine_attempts)
                return outputs, True
            log.info("[SelfRefine] Attempt %d/%d — user did not pass. Feedback: %s", attempt_num, max_refine_attempts, feedback)
            if attempt_num < max_refine_attempts:
                feedback, prompt_override = get_user_feedback_interactive(
                    feedback, score, llm_passed, task_name, log
                )
                log.info("[SelfRefine] Refining artifact for next attempt ...")
                artifact = refine_fn(artifact, outputs, feedback, refine_prompt_override=prompt_override)
        log.warning("[SelfRefine] Exhausted %d attempts without user pass.", max_refine_attempts)
        final_task_def = history[-1].outputs if history else None
        return final_task_def, False
    else:
        # SelfRefineLoop expects evaluate_fn to return (passed, feedback) only
        evaluate_fn_for_loop = lambda outputs: evaluate_fn(outputs)[:2]
        loop = SelfRefineLoop(max_attempts=max_refine_attempts, logger=log)
        result = loop.run(
            initial_artifact=None,
            generate_fn=generate_fn,
            evaluate_fn=evaluate_fn_for_loop,
            refine_fn=lambda a, o, f: refine_fn(a, o, f, None),
        )
        final_task_def = result.history[-1].outputs if result.history else None
        return final_task_def, result.passed


# ---------------------------------------------------------------------------
# Top-level entry point (with caching)
# ---------------------------------------------------------------------------


def extract_highlevel_task(config: HighLevelConfig) -> Dict:
    """Extract and cache a high-level task definition for a given corpus.

    If a cached result exists for the same examples, it is returned immediately
    unless ``config.force_generate`` is True.

    Parameters
    ----------
    config : HighLevelConfig

    Returns
    -------
    dict with keys: task_name, query_type, doc_type, relevance_definition,
                    passed, from_cache, cache_path.
    """
    # Setup logging
    os.makedirs(config.log_dir, exist_ok=True)
    log_file = os.path.join(
        config.log_dir,
        f"HighLevel_Generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger = logging.getLogger(f"HighLevel.{config.task_name}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler())

    task_name = config.task_name.lower()

    # Retrieve examples for the target corpus
    examples = get_examples(task_name)
    if not examples:
        # Try the raw dict in case the key exists but get_examples normalises differently
        examples = Few_Shot_Example.get(task_name, [])
    if not examples:
        raise ValueError(
            f"No few-shot examples found for task '{task_name}'. "
            f"Available tasks: {list(Few_Shot_Example.keys())}"
        )

    logger.info("Loaded %d few-shot examples for task '%s'.", len(examples), task_name)

    # ---- Cache check ---------------------------------------------------------
    examples_hash = hashlib.md5(
        json.dumps(examples, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    cache_dir = Path(__file__).parent / config.output_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_files = sorted(
        cache_dir.glob(f"HighLevel_{task_name}_{examples_hash}_*.json"),
        reverse=True,
    )

    if cache_files and not config.force_generate:
        for cached_path in cache_files:
            with open(cached_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_demo_max_examples = int(cached.get("demo_max_examples", 3))
            if cached_demo_max_examples == config.demo_max_examples:
                logger.info("Using cached definition for task '%s': %s", task_name, cached_path)
                cached["from_cache"] = True
                cached["cache_path"] = str(cached_path)
                return cached
        logger.info(
            "Found cache files for task '%s', but none matched demo_max_examples=%d.",
            task_name,
            config.demo_max_examples,
        )

    # ---- Generate ------------------------------------------------------------
    llm_cfg = config.get_llm_cfg()
    logger.info("LLM config: model=%s", llm_cfg["model"])

    task_def, passed = generate_highlevel_task(
        task_name=task_name,
        examples=examples,
        llm_cfg=llm_cfg,
        max_refine_attempts=config.max_refine_attempts,
        demo_max_examples=config.demo_max_examples,
        enable_user_feedback=config.enable_user_feedback,
        logger=logger,
    )

    if task_def is None:
        raise RuntimeError(f"Failed to generate a high-level task definition for '{task_name}'.")

    # ---- Persist -------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = cache_dir / f"HighLevel_{task_name}_{examples_hash}_{timestamp}.json"
    output = {
        "task_name": task_name,
        "query_type": task_def.get("query_type", ""),
        "doc_type": task_def.get("doc_type", ""),
        "relevance_definition": task_def.get("relevance_definition", ""),
        "demo_max_examples": config.demo_max_examples,
        "passed": passed,
        "from_cache": False,
        "cache_path": str(save_path),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Definition saved to: %s", save_path)

    _log_result(output, logger)
    return output


def _log_result(output: Dict, logger: logging.Logger) -> None:
    """Pretty-print the generated definition."""
    logger.info("=" * 60)
    logger.info("Generated High-Level Task Definition for '%s':", output["task_name"])
    logger.info("  query_type         : %s", output["query_type"])
    logger.info("  doc_type           : %s", output["doc_type"])
    logger.info("  relevance_definition: %s", output["relevance_definition"])
    logger.info("  self-refine passed : %s", output["passed"])
    logger.info("=" * 60)
    # Emit the Python tuple suitable for copy-pasting into HighLevel_Def.py
    logger.info(
        'Ready to add to HighLevel_Def.py:\n    "%s": (\n        "%s",\n        "%s",\n        "%s"\n    ),',
        output["task_name"],
        output["query_type"],
        output["doc_type"],
        output["relevance_definition"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> HighLevelConfig:
    p = argparse.ArgumentParser(
        description="Infer a high-level IR task definition from few-shot examples."
    )
    p.add_argument(
        "--task", type=str, required=True, dest="task_name",
        help="Task name matching a key in Few_Shot_Example (e.g. trec-covid).",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="LLM model name (default: gpt-4o-mini).",
    )
    p.add_argument(
        "--model-server", type=str, default=None, dest="model_server",
        help="LLM API base URL (overrides OPENAI_BASE_URL env var).",
    )
    p.add_argument(
        "--api-key", type=str, default=None, dest="api_key",
        help="LLM API key (overrides OPENAI_API_KEY env var).",
    )
    p.add_argument(
        "--max-refine", type=int, default=3, dest="max_refine_attempts",
        help="Maximum self-refine attempts (default: 3).",
    )
    p.add_argument(
        "--demo-max-examples", type=int, default=3, dest="demo_max_examples",
        help="Maximum examples shown in each reference demonstration block (default: 3).",
    )
    p.add_argument(
        "--force", action="store_true", dest="force_generate",
        help="Bypass cache and always call the LLM.",
    )
    p.add_argument(
        "--output-dir", type=str, default="Results/Instructions", dest="output_dir",
        help='Cache directory (default: Results/Instructions).',
    )
    p.add_argument(
        "--log-dir", type=str, default="Logs", dest="log_dir",
    )
    p.add_argument(
        "--interactive", action="store_true", dest="enable_user_feedback",
        help="After each failed evaluation, prompt for editing feedback and optional custom refine prompt (via input()).",
    )

    args = p.parse_args()
    cfg = HighLevelConfig(
        task_name=args.task_name,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        max_refine_attempts=args.max_refine_attempts,
        demo_max_examples=max(1, args.demo_max_examples),
        force_generate=args.force_generate,
        enable_user_feedback=args.enable_user_feedback,
    )
    if args.model:
        cfg.model = args.model
    if args.model_server:
        cfg.model_server = args.model_server
    if args.api_key:
        cfg.api_key = args.api_key
    return cfg


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = parse_args()
    result = extract_highlevel_task(cfg)

    print("\n--- Result ---")
    print(json.dumps(
        {k: v for k, v in result.items() if k != "from_cache"},
        ensure_ascii=False,
        indent=2,
    ))

    # Print the copy-paste snippet for HighLevel_Def.py
    print("\n--- Add to HighLevel_Def.py ---")
    print(f'    "{result["task_name"]}": (')
    print(f'        "{result["query_type"]}",')
    print(f'        "{result["doc_type"]}",')
    print(f'        "{result["relevance_definition"]}"')
    print("    ),")
