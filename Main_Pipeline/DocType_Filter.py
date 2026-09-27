import json
import json5
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from concurrent_runner import run_concurrent

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

from Few_Shot_Example import Few_Shot_Example
from HighLevel_Def import HighLevel_Task_Definition

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"Results/Logs/DocType_Filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

@dataclass
class FilterConfig:
    input_path: str
    task_name: str
    output_dir: str
    log_dir: str
    batch_size: int = 10
    model: str = "gpt-4o-mini"
    model_server: Optional[str] = None
    api_key: Optional[str] = None
    # Concurrency
    concurrency_enabled: bool = False
    max_workers: Optional[int] = None
    jitter: float = 0.0
    filter_inject_pos_examples: bool = False
    filter_pos_examples_count: int = 3
    filter_pos_example_max_chars: int = 1200


def _normalize_prompt_text(text: str) -> str:
    return " ".join((text or "").split())


def _truncate_for_prompt(text: str, max_chars: int) -> str:
    text = _normalize_prompt_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = " ... "
    if max_chars <= len(marker) + 2:
        return text[:max_chars]
    available = max_chars - len(marker)
    head_chars = max(1, int(available * 0.7))
    tail_chars = max(1, available - head_chars)
    return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()


def build_positive_reference_block(
    task_name: str,
    max_examples: int = 3,
    max_chars: int = 1200,
) -> tuple[str, int]:
    """Build an optional positive-only reference block for doc-type filtering."""
    if max_examples <= 0:
        return "", 0

    examples = Few_Shot_Example.get(task_name) or []
    rendered_examples = []

    for example in examples:
        positive = _truncate_for_prompt(example.get("Positive", ""), max_chars)
        if not positive:
            continue
        rendered_examples.append(
            f"### Positive Prototype {len(rendered_examples) + 1}\n"
            f"{positive}"
        )
        if len(rendered_examples) >= max_examples:
            break

    if not rendered_examples:
        return "", 0

    block = """## Reference Positive Prototypes
The following texts are known positive examples from the same task. Use them only as prototypes to internalize what a genuine document of the target type typically looks like.
- Learn from their genre, discourse style, topical granularity, and overall writing pattern
- Do NOT require the candidate document to match the same topic, wording, or structure
- Do NOT switch into query relevance judging; these examples are only here to sharpen document-type recognition
- If the examples and the rubric ever appear to conflict, follow the rubric and the document-type definition

""" + "\n\n".join(rendered_examples)
    return block, len(rendered_examples)


def build_filter_prompt(
    docs_batch: List[dict],
    doc_type: str,
    relevance_def: str,
    positive_reference_block: str = "",
) -> str:
    """Build a prompt that asks the LLM to score each doc 0-3 for doc-type fit.

    Parameters
    ----------
    docs_batch : list of dicts with keys doc_id, doc
    doc_type   : expected document type string from HighLevel_Task_Definition
    relevance_def : relevance definition string (provides extra context about doc purpose)
    """
    doc_lines = []
    for i, doc in enumerate(docs_batch, 1):
        doc_content = (doc.get("doc") or "").strip()
        doc_lines.append(
            f"[{i}] doc_id: \"{doc['doc_id']}\"\n"
            + f"    Doc: {doc_content}"
        )
    docs_block = "\n\n".join(doc_lines)
    optional_reference_section = f"\n\n{positive_reference_block}" if positive_reference_block else ""

    if optional_reference_section == "":
        return f"""You are an expert document curator specializing in information retrieval datasets. Your task is to evaluate whether each document genuinely belongs to a specific document type.

## Task Context
- **Document Type**: {doc_type}
- **Relevance Definition**: {relevance_def}

## Evaluation Protocol
1. **Independent assessment**: Evaluate each document in isolation. Do NOT compare documents against each other.
2. **Focus on document type**: Your primary goal is to verify whether the document IS a genuine "{doc_type}", not whether it's relevant to any specific query.
3. **Consider the domain context**: Use the relevance definition above to understand the domain and purpose of these documents.

## Scoring Rubric (0-3 Scale)

### Score 3 — Perfect Match (PASS)
The document is an **exemplary, genuine instance** of "{doc_type}".
- Content, structure, and style fully align with what "{doc_type}" should be
- No significant deviations from the expected document type characteristics
- Would serve as a good example of this document type in a dataset

### Score 2 — Good Match (PASS)
The document **belongs to "{doc_type}"** but may have minor issues.
- Core content and purpose align with "{doc_type}"
- Minor deviations in format, completeness, or presentation do not change its fundamental nature
- Still suitable as a valid example of this document type

### Score 1 — Borderline / Weak Match (FAIL)
The document has **some superficial relevance** to "{doc_type}" but does NOT truly belong.
- May share some keywords, topics, or surface characteristics
- Wrong genre, format, or purpose (e.g., a discussion about "{doc_type}" rather than being one)
- Partially related content that doesn't satisfy the core definition of "{doc_type}"
- Mixed-language content where the expected language is not dominant

### Score 0 — Clear Mismatch (FAIL)
The document is **completely off-topic** or clearly wrong type.
- Entirely different document genre or format
- Wrong language entirely
- No meaningful connection to "{doc_type}" whatsoever

## Pass/Fail Threshold
- **PASS** (include in dataset): Score 2 or 3
- **FAIL** (filter out): Score 0 or 1

## Documents to Evaluate

{docs_block}

## Output Format (valid JSON only, no Markdown fences)
{{
  "results": [
    {{
      "doc_id": "...",
      "score": 0-3,
      "verdict": "pass" | "fail",
      "reason": "Brief explanation citing specific characteristics that led to this score"
    }}
  ]
}}

Requirements:
- Include exactly one entry per document in the same order
- Score must be an integer 0-3
- Verdict must be "pass" for scores 2-3, "fail" for scores 0-1
- Reason should be specific and reference actual content from the document
- Output ONLY the JSON object"""

    return f"""You are an expert document curator specializing in information retrieval datasets. Your task is to evaluate whether each document genuinely belongs to a specific document type.

## Task Context
- **Document Type**: {doc_type}
- **Relevance Definition**: {relevance_def}
{optional_reference_section}

## Evaluation Protocol
1. **Independent assessment**: Evaluate each document in isolation. Do NOT compare documents against each other.
2. **Focus on document type**: Your primary goal is to verify whether the document IS a genuine "{doc_type}", not whether it's relevant to any specific query.
3. **Consider the domain context**: Use the relevance definition above to understand the domain and purpose of these documents.
4. **Use the prototypes carefully**: Use the positive prototypes above to calibrate the typical style and substance of a true "{doc_type}" document, not as a rigid checklist.
5. **Prioritize genre over keywords**: A document can mention the right topic yet still fail if its genre, purpose, or discourse function is wrong.

## Scoring Rubric (0-3 Scale)

### Score 3 — Perfect Match (PASS)
The document is an **exemplary, genuine instance** of "{doc_type}".
- Content, structure, and style fully align with what "{doc_type}" should be
- No significant deviations from the expected document type characteristics
- Would serve as a good example of this document type in a dataset

### Score 2 — Good Match (PASS)
The document **belongs to "{doc_type}"** but may have minor issues.
- Core content and purpose align with "{doc_type}"
- Minor deviations in format, completeness, or presentation do not change its fundamental nature
- Still suitable as a valid example of this document type

### Score 1 — Borderline / Weak Match (FAIL)
The document has **some superficial relevance** to "{doc_type}" but does NOT truly belong.
- May share some keywords, topics, or surface characteristics
- Wrong genre, format, or purpose (e.g., a discussion about "{doc_type}" rather than being one)
- Partially related content that doesn't satisfy the core definition of "{doc_type}"
- Mixed-language content where the expected language is not dominant
- Feels materially weaker, noisier, or less type-faithful than the positive prototypes

### Score 0 — Clear Mismatch (FAIL)
The document is **completely off-topic** or clearly wrong type.
- Entirely different document genre or format
- Wrong language entirely
- No meaningful connection to "{doc_type}" whatsoever

## Pass/Fail Threshold
- **PASS** (include in dataset): Score 2 or 3
- **FAIL** (filter out): Score 0 or 1

## Documents to Evaluate

{docs_block}

## Output Format (valid JSON only, no Markdown fences)
{{
  "results": [
    {{
      "doc_id": "...",
      "score": 0-3,
      "verdict": "pass" | "fail",
      "reason": "Brief explanation citing specific characteristics that led to this score"
    }}
  ]
}}

Requirements:
- Include exactly one entry per document in the same order
- Score must be an integer 0-3
- Verdict must be "pass" for scores 2-3, "fail" for scores 0-1
- Reason should be specific and reference actual content from the document
- Output ONLY the JSON object"""


def parse_filter_response(raw_text: str, docs_batch: List[dict], logger=None) -> List[dict]:
    """Parse the LLM filter response (with 0-3 scores) into a list of result dicts.

    Falls back to failing all docs in the batch if parsing fails, to avoid
    accidentally keeping mismatched data due to an LLM formatting error.

    Score interpretation:
    - 3: Perfect match (pass)
    - 2: Good match (pass)
    - 1: Borderline/weak match (fail)
    - 0: Clear mismatch (fail)
    """
    log = logger or logging.getLogger(__name__)
    try:
        parsed = json5.loads(raw_text)
        results = parsed.get("results", [])
        if not isinstance(results, list) or len(results) == 0:
            raise ValueError("Empty or missing 'results' list")

        # Normalize verdict based on score (score >= 2 is pass, < 2 is fail)
        for r in results:
            score = r.get("score")
            # If score is provided, use it to determine verdict
            if score is not None:
                try:
                    score = int(score)
                    r["score"] = score
                    # Verdict: pass if score >= 2, fail otherwise
                    r["verdict"] = "pass" if score >= 2 else "fail"
                except (ValueError, TypeError):
                    # If score is not a valid integer, fall back to explicit verdict
                    r["verdict"] = str(r.get("verdict", "pass")).strip().lower()
            else:
                # No score provided, use explicit verdict
                r["verdict"] = str(r.get("verdict", "pass")).strip().lower()

        return results
    except Exception as exc:
        log.warning(
            "Failed to parse filter response (%s); defaulting all %d docs to 'fail'. "
            "Raw output (first 500 chars): %s",
            exc, len(docs_batch), raw_text[:500],
        )
        return [{"doc_id": d["doc_id"], "verdict": "fail", "score": 0, "reason": "parse error — defaulted to fail"} for d in docs_batch]


@register_tool('doctype_filter')
class DiverseTextFilter(BaseTool):
    description = (
        "Filter a JSONL file of diverse texts, keeping only documents whose content "
        "matches the expected document type for the given task. Uses an LLM to make "
        "binary pass/fail decisions, processing documents in configurable batches."
    )
    parameters = [
        {
            'name': 'input_path',
            'type': 'string',
            'description': 'Path to the input JSONL file (lines with doc_id, title, text).',
            'required': True,
        },
        {
            'name': 'task_name',
            'type': 'string',
            'description': 'Task name present in HighLevel_Task_Definition.',
            'required': True,
        },
        {
            'name': 'output_path',
            'type': 'string',
            'description': 'Path to write the filtered JSONL output.',
            'required': True,
        },
        {
            'name': 'batch_size',
            'type': 'integer',
            'description': 'Number of documents to send to the LLM in a single call.',
            'required': False,
        },
        {
            'name': 'filter_inject_pos_examples',
            'type': 'boolean',
            'description': 'Whether to inject few-shot positive examples into the filter prompt.',
            'required': False,
        },
        {
            'name': 'filter_pos_examples_count',
            'type': 'integer',
            'description': 'Maximum number of positive examples to inject into the filter prompt.',
            'required': False,
        },
        {
            'name': 'filter_pos_example_max_chars',
            'type': 'integer',
            'description': 'Maximum characters kept for each injected positive example.',
            'required': False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        params_dict = json5.loads(params)
        input_path = params_dict["input_path"]
        task_name = params_dict["task_name"]
        output_path = params_dict["output_path"]
        batch_size = int(params_dict.get("batch_size", 10))
        filter_inject_pos_examples = bool(params_dict.get("filter_inject_pos_examples", False))
        filter_pos_examples_count = max(0, int(params_dict.get("filter_pos_examples_count", 3)))
        filter_pos_example_max_chars = max(200, int(params_dict.get("filter_pos_example_max_chars", 1200)))

        logger = kwargs.get("logger") or logging.getLogger(__name__)

        if task_name not in HighLevel_Task_Definition:
            return json5.dumps({"error": f"Task '{task_name}' not found in HighLevel_Task_Definition."})

        _query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]

        llm_cfg = kwargs.get("llm_cfg", {
            "model": "gpt-4.1-mini",
            "model_server": os.getenv("OPENAI_BASE_URL"),
            "api_key": os.getenv("OPENAI_API_KEY"),
        })
        conc_enabled = bool(kwargs.get("concurrency_enabled", False))
        conc_max_workers = kwargs.get("max_workers", None)
        conc_jitter = float(kwargs.get("jitter", 0.0))

        # Read all documents from the input JSONL
        docs: List[dict] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))

        total_docs = len(docs)
        logger.info(
            "DocType filter: task=%s doc_type='%s' total_docs=%d batch_size=%d inject_pos_examples=%s",
            task_name, doc_type, total_docs, batch_size, filter_inject_pos_examples,
        )

        positive_reference_block = ""
        if filter_inject_pos_examples:
            positive_reference_block, injected_count = build_positive_reference_block(
                task_name,
                max_examples=filter_pos_examples_count,
                max_chars=filter_pos_example_max_chars,
            )
            if injected_count:
                logger.info(
                    "DocType filter: injected %d positive few-shot example(s) into prompt (max_chars=%d).",
                    injected_count, filter_pos_example_max_chars,
                )
            else:
                logger.warning(
                    "DocType filter: filter_inject_pos_examples=true but no usable positive few-shot examples found for task=%s.",
                    task_name,
                )

        passed_docs: List[dict] = []
        filtered_count = 0

        # Build per-batch work items
        batches = [
            docs[batch_start: batch_start + batch_size]
            for batch_start in range(0, total_docs, batch_size)
        ]
        total_batches = len(batches)
        logger.info(
            "DocType filter: %d batches (concurrency=%s, max_workers=%s, jitter=%.2f)",
            total_batches, conc_enabled, conc_max_workers, conc_jitter,
        )

        def _filter_one_batch(batch: List[dict], batch_num: int) -> dict:
            """Run a single LLM filter call and return verdict_map for the batch.

            On any exception, defaults all docs in the batch to 'fail', matching
            ``parse_filter_response``'s fallback behavior.
            """
            logger.info(
                "DocType filter: batch %d/%d (docs %d-%d)",
                batch_num, total_batches,
                (batch_num - 1) * batch_size + 1,
                (batch_num - 1) * batch_size + len(batch),
            )
            try:
                prompt = build_filter_prompt(
                    batch,
                    doc_type,
                    relevance_def,
                    positive_reference_block=positive_reference_block,
                )
                agent = Assistant(llm=llm_cfg)
                raw = ""
                for response in agent.run([{"role": "user", "content": prompt}]):
                    raw = response[-1]["content"]
                results = parse_filter_response(raw, batch, logger=logger)
                # Return both verdict and score for richer logging
                return {
                    r["doc_id"]: {
                        "verdict": r.get("verdict", "fail"),
                        "score": r.get("score", 0),
                        "reason": r.get("reason", "")
                    }
                    for r in results
                }
            except Exception as exc:
                logger.error(
                    "DocType filter: batch %d/%d failed (%s); defaulting all %d docs to 'fail'.",
                    batch_num, total_batches, exc, len(batch),
                )
                return {
                    doc["doc_id"]: {"verdict": "fail", "score": 0, "reason": "batch processing error"}
                    for doc in batch
                }

        batch_kwargs = [
            {"batch": batch, "batch_num": i + 1}
            for i, batch in enumerate(batches)
        ]
        verdict_maps = run_concurrent(
            _filter_one_batch,
            batch_kwargs,
            enabled=conc_enabled,
            max_workers=conc_max_workers,
            jitter=conc_jitter,
        )

        # Track score distribution for analytics
        score_distribution = {0: 0, 1: 0, 2: 0, 3: 0}

        for batch, verdict_map in zip(batches, verdict_maps):
            for doc in batch:
                result = verdict_map.get(doc["doc_id"], {"verdict": "fail", "score": 0, "reason": "missing verdict — defaulted to fail"})
                verdict = result.get("verdict", "pass")
                score = result.get("score", 2 if verdict == "pass" else 1)
                reason = result.get("reason", "")

                # Track score distribution
                if isinstance(score, int) and 0 <= score <= 3:
                    score_distribution[score] = score_distribution.get(score, 0) + 1

                if verdict == "pass":
                    passed_docs.append(doc)
                else:
                    filtered_count += 1
                    logger.debug(
                        "DocType filter: FILTERED doc_id=%s score=%s reason=%s",
                        doc["doc_id"], score, reason[:100]  # Truncate long reasons
                    )

        # Write filtered docs to output JSONL
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in passed_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        passed_count = len(passed_docs)
        # Calculate detailed stats from score distribution
        perfect_match = score_distribution.get(3, 0)
        good_match = score_distribution.get(2, 0)
        borderline = score_distribution.get(1, 0)
        clear_mismatch = score_distribution.get(0, 0)

        logger.info(
            "DocType filter done: total=%d passed=%d filtered=%d → output=%s",
            total_docs, passed_count, filtered_count, output_path,
        )
        logger.info(
            "Score distribution: perfect_match(score_3)=%d good_match(score_2)=%d "
            "borderline(score_1)=%d clear_mismatch(score_0)=%d",
            perfect_match, good_match, borderline, clear_mismatch,
        )

        return json5.dumps({
            "output_path": output_path,
            "total_docs": total_docs,
            "passed": passed_count,
            "filtered": filtered_count,
            "score_distribution": {
                "score_3_perfect_match": perfect_match,
                "score_2_good_match": good_match,
                "score_1_borderline": borderline,
                "score_0_clear_mismatch": clear_mismatch,
            },
        }, ensure_ascii=False)


def filter_diverse_texts(config: FilterConfig, logger=None) -> str:
    """Run doc-type filtering on a diverse texts JSONL file.

    Mirrors the interface of ``select_diverse_texts``: accepts a config object,
    returns the path of the filtered output file.

    Parameters
    ----------
    config : FilterConfig
    logger : logging.Logger, optional

    Returns
    -------
    str
        Absolute path to the filtered JSONL file.
    """
    log = logger or logging.getLogger(__name__)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.output_dir, f"filtered_diverse_texts_{timestamp}.jsonl")

    llm_cfg = {
        "model": config.model,
        "model_server": config.model_server or os.getenv("OPENAI_BASE_URL"),
        "api_key": config.api_key or os.getenv("OPENAI_API_KEY"),
    }

    tool = DiverseTextFilter()
    result_json = tool.call(
        json5.dumps({
            "input_path": config.input_path,
            "task_name": config.task_name,
            "output_path": output_path,
            "batch_size": config.batch_size,
            "filter_inject_pos_examples": config.filter_inject_pos_examples,
            "filter_pos_examples_count": config.filter_pos_examples_count,
            "filter_pos_example_max_chars": config.filter_pos_example_max_chars,
        }),
        llm_cfg=llm_cfg,
        logger=log,
        concurrency_enabled=config.concurrency_enabled,
        max_workers=config.max_workers,
        jitter=config.jitter,
    )

    result = json5.loads(result_json)
    if "error" in result:
        raise RuntimeError(f"DocType filter failed: {result['error']}")

    log.info(
        "=== Step 2.5 done === output=%s total=%d passed=%d filtered=%d",
        result["output_path"], result["total_docs"], result["passed"], result["filtered"],
    )
    return result["output_path"]


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    llm_cfg = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "model_server": os.getenv("OPENAI_BASE_URL"),
        "api_key": os.getenv("OPENAI_API_KEY"),
    }

    task = sys.argv[1] if len(sys.argv) > 1 else "arguana"
    input_file = sys.argv[2] if len(sys.argv) > 2 else None

    if input_file is None:
        logging.error("Usage: python DocType_Filter.py <task_name> <input_jsonl_path>")
        sys.exit(1)

    cfg = FilterConfig(
        input_path=input_file,
        task_name=task,
        output_dir="Results",
        log_dir="Results/Logs",
        batch_size=5,
        model=llm_cfg["model"],
        model_server=llm_cfg["model_server"],
        api_key=llm_cfg["api_key"],
        filter_inject_pos_examples=False,
    )

    os.makedirs("Results", exist_ok=True)
    os.makedirs("Results/Logs", exist_ok=True)

    out = filter_diverse_texts(cfg)
    logging.info("Filtered output: %s", out)
