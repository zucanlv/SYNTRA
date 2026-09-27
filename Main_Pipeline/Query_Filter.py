"""LLM-based per-query style filter for Step-3 DiverseQuery JSON.

在 ``Main_Pipeline`` 目录下执行；把所有 ``<...>`` 换成你的值。

下面是一条**写出全部 CLI 参数**的模板（复制到终端后改占位符）：

- ``--query-instruction`` 与 ``--query-instruction-file`` **二选一**：保留其一，删掉另一行。
- ``--inject-golden-queries`` 与 ``--no-inject-golden-queries`` **二选一**（下面示例用前者）。
- ``--concurrency`` 与 ``--no-concurrency`` **二选一**（下面示例用 ``--no-concurrency``；若开并发可改成 ``--concurrency`` 并填 ``--max-workers``）。
- ``--model`` / ``--model-server`` / ``--api-key`` 可整行删掉，改用环境变量 ``OPENAI_MODEL``、``OPENAI_BASE_URL``、``OPENAI_API_KEY``。

::

    python Query_Filter.py \
        --task msmarco \
        --input /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/msmarco_20260418_092016/DiverseQuery_Main_20260418_103048.json \
        --query-instruction-file /data/share/project/shared_datasets/DSA/others/Instructions/msmarco/QueryInstrRefine_msmarco_20260327_084310.txt \
        --output-dir /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/msmarco_20260418_092016 \
        --batch-size 4 \
        --model /data/share/project/shared_models/Qwen3-30B-A3B-Instruct-2507 \
        --model-server http://127.0.0.1:8000/v1 \
        --api-key EMPTY \
        --inject-golden-queries \
        --golden-query-count 3 \
        --golden-query-max-chars 1000 \
        --concurrency \
        --max-workers 8 \
        --jitter 0.3

若不用文件、改为内联风格说明：删掉 ``--query-instruction-file ...`` 这一行，改为
``--query-instruction '<整段文字>'``（注意 shell 引号）。

其它布尔开关（与上表**二选一**，不要两行同时出现）：

- Golden：``--inject-golden-queries`` **或** ``--no-inject-golden-queries``
- 并发：``--no-concurrency`` **或** ``--concurrency``（仅后者生效时 ``--max-workers`` / ``--jitter`` 才有意义）

短选项等价：``-t`` = ``--task``，``-i`` = ``--input``，``-o`` = ``--output-dir``。

python Query_Filter.py \
        --task climate-fever \
        --input /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/climate-fever_20260421_100118/DiverseQuery_Main_20260421_105957.json \
        --query-instruction-file /data/share/project/shared_datasets/DSA/others/Instructions/climate-fever/QueryInstrRefine_climate-fever_20260421_093739.txt \
        --output-dir /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/climate-fever_20260421_100118 \
        --batch-size 4 \
        --model /data/share/project/shared_models/Qwen3-30B-A3B-Instruct-2507 \
        --model-server http://127.0.0.1:8000/v1 \
        --api-key EMPTY \
        --inject-golden-queries \
        --golden-query-count 3 \
        --golden-query-max-chars 1000 \
        --concurrency \
        --max-workers 8 \
        --jitter 0.3

python Query_Filter.py \
        --task dbpedia \
        --input /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/dbpedia_20260422_124344/DiverseQuery_Main_20260422_135622.json \
        --query-instruction-file /data/share/project/shared_datasets/DSA/others/Instructions/dbpedia/QueryInstrRefine_dbpedia_20260422_123753.txt \
        --output-dir /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/dbpedia_20260422_124344 \
        --batch-size 4 \
        --model /data/share/project/shared_models/Qwen3-30B-A3B-Instruct-2507 \
        --model-server http://127.0.0.1:8000/v1 \
        --api-key EMPTY \
        --inject-golden-queries \
        --golden-query-count 3 \
        --golden-query-max-chars 1000 \
        --concurrency \
        --max-workers 8 \
        --jitter 0.3

python Query_Filter.py \
        --task touche2020 \
        --input /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/touche2020_20260423_002611/DiverseQuery_Main_20260423_014141.json \
        --query-instruction-file /data/share/project/shared_datasets/DSA/others/Instructions/touche2020/QueryInstrRefine_touche2020_20260423_001330.txt \
        --output-dir /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Results/touche2020_20260423_002611 \
        --batch-size 4 \
        --model /data/share/project/shared_models/Qwen3-30B-A3B-Instruct-2507 \
        --model-server http://127.0.0.1:8000/v1 \
        --api-key EMPTY \
        --inject-golden-queries \
        --golden-query-count 3 \
        --golden-query-max-chars 1000 \
        --concurrency \
        --max-workers 8 \
        --jitter 0.3

"""

import argparse
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
        logging.FileHandler(f"Results/Logs/Query_Filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)


@dataclass
class QueryFilterConfig:
    input_path: str           # Path to DiverseQuery_Main_*.json
    task_name: str
    query_instruction: str    # Loaded QueryInstrRefine_*.txt content (style guide)
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
    inject_golden_queries: bool = True
    golden_query_count: int = 3
    golden_query_max_chars: int = 200


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


def build_golden_query_block(
    task_name: str,
    max_examples: int = 3,
    max_chars: int = 200,
) -> tuple[str, int]:
    """Build a golden-query reference block for query-style filtering.

    Uses the 'query' field from Few_Shot_Example (not 'Positive'), giving the
    LLM concrete style examples of what a correct query looks like.
    """
    if max_examples <= 0:
        return "", 0

    examples = Few_Shot_Example.get(task_name) or []
    rendered = []

    for example in examples:
        query_text = _truncate_for_prompt(example.get("query", ""), max_chars)
        if not query_text:
            continue
        rendered.append(
            f"### Golden Example {len(rendered) + 1}\n"
            f"{query_text}"
        )
        if len(rendered) >= max_examples:
            break

    if not rendered:
        return "", 0

    block = (
        "## Golden Query Examples (style benchmark)\n"
        "The following queries are known high-quality examples for this task. "
        "- These examples demonstrate the *style* expected for all generated queries\n"
        "- A generated query must feel qualitatively similar to these examples\n"
        "- **Important: try your best to feel the sense of these golden queries, and evaluate the generated query based on the sense**\n"
        + "\n\n".join(rendered)
    )
    return block, len(rendered)


def build_query_filter_prompt(
    queries_batch: List[dict],
    query_type: str,
    relevance_def: str,
    query_instruction: str,
    golden_query_block: str = "",
) -> str:
    """Build a prompt for per-query style/quality filtering.

    Parameters
    ----------
    queries_batch : list of dicts with keys query_id (e.g. 'q1') and query_text
    query_type    : expected query type string from HighLevel_Task_Definition
    relevance_def : relevance definition string (task context)
    query_instruction : loaded QueryInstrRefine_*.txt style guide
    golden_query_block : formatted golden-query reference block (may be empty)
    """
    query_lines = []
    for item in queries_batch:
        query_lines.append(
            f"[{item['query_id']}] \"{item['query_text']}\""
        )
    queries_block = "\n".join(query_lines)

    optional_golden_section = f"\n\n{golden_query_block}" if golden_query_block else ""
    optional_style_section = (
        f"\n\n## Query Style Guidelines\n{query_instruction}"
        if query_instruction and query_instruction.strip()
        else ""
    )

    return f"""You are an expert query curator for an information retrieval dataset synthesis pipeline. \
Your task is to evaluate whether each generated query meets the required quality standard for this task.

## Task Context
- **Query Type**: {query_type}
- **Relevance Definition**: {relevance_def}{optional_style_section}{optional_golden_section}

## Evaluation Protocol
1. **Independent assessment**: Evaluate each query in isolation. Do NOT compare queries against each other.
2. **Focus on style and query type**: Your primary goal is to verify whether each query is a genuine \
"{query_type}", not whether it is relevant to any specific document. To think the question 'Is this a genuine "{query_type}"?'
3. **Cross-check style**: Use the style guidelines as explicit rules and the golden examples \
(if provided) as concrete demonstrations of the expected style — both must be satisfied.

## Scoring Rubric (0-3 Scale)

### Score 3 — Perfect Style Match (PASS)
The query **fully adheres** to the style guidelines and its sense is indistinguishable from a genuine "{query_type}" query.
- Correct format, length, vocabulary register, and sentence structure
- **Sounds natural as a {query_type}; no templated or artificial phrasing**
- Every constraint in the style guidelines is satisfied

### Score 2 — Good Style Match (PASS)
The query **broadly follows** the style guidelines and its sense is similar to a genuine "{query_type}" query.
- Minor isolated deviations (e.g., slightly longer, marginally different phrasing) that do not undermine its nature as a "{query_type}"
- Core style requirements are met; only small refinements would close the gap

### Score 1 — Borderline / Weak Match (FAIL)
The query **deviates noticeably** from the style guidelines.
- Violates one or more explicit style constraints (e.g., wrong interrogative form, forbidden phrasing, wrong length range)
- Feels materially different in register or structure from the golden examples
- Would require meaningful rewriting to qualify as a proper "{query_type}"

### Score 0 — Clear Mismatch (FAIL)
The query **fundamentally fails** to match the expected query type.
- Describes or narrates a document rather than asking a question
- Targets a completely different query type (e.g., factoid question when argumentative is required)
- Incoherent, garbled, or empty

## Pass/Fail Threshold
- **PASS** (include in dataset): Score 2 or 3
- **FAIL** (filter out): Score 0 or 1

## Queries to Evaluate

{queries_block}

## Output Format (valid JSON only, no Markdown fences)
{{
  "results": [
    {{
      "query_id": "q1",
      "score": 0-3,
      "verdict": "pass" | "fail",
      "reason": "Brief explanation citing the specific characteristic that led to this score"
    }}
  ]
}}

Requirements:
- Include exactly one entry per query in the same order as presented
- Score must be an integer 0-3
- Verdict must be "pass" for scores 2-3, "fail" for scores 0-1
- Reason should be specific (e.g., cite the exact rule violated or the style mismatch observed)
- Output ONLY the JSON object"""


def parse_query_filter_response(
    raw_text: str,
    batch_query_ids: List[str],
    logger=None,
) -> dict:
    """Parse the LLM query-filter response into a verdict map keyed by query_id.

    Falls back to failing all queries in the batch if parsing fails, matching
    DocType_Filter's conservative-failure approach.

    Returns
    -------
    dict mapping query_id -> {"verdict": "pass"|"fail", "score": int, "reason": str}
    """
    log = logger or logging.getLogger(__name__)
    try:
        parsed = json5.loads(raw_text)
        results = parsed.get("results", [])
        if not isinstance(results, list) or len(results) == 0:
            raise ValueError("Empty or missing 'results' list")

        verdict_map = {}
        for r in results:
            qid = str(r.get("query_id", "")).strip()
            score = r.get("score")
            try:
                score = int(score)
            except (ValueError, TypeError):
                score = 0
            verdict = "pass" if score >= 2 else "fail"
            verdict_map[qid] = {
                "verdict": verdict,
                "score": score,
                "reason": r.get("reason", ""),
            }

        # Fill any missing query_ids with conservative fail
        for qid in batch_query_ids:
            if qid not in verdict_map:
                log.warning(
                    "Query filter: query_id '%s' missing from LLM response; defaulting to 'fail'.",
                    qid,
                )
                verdict_map[qid] = {"verdict": "fail", "score": 0, "reason": "missing in response — defaulted to fail"}

        return verdict_map

    except Exception as exc:
        log.warning(
            "Failed to parse query filter response (%s); defaulting all %d queries to 'fail'. "
            "Raw output (first 500 chars): %s",
            exc, len(batch_query_ids), raw_text[:500],
        )
        return {
            qid: {"verdict": "fail", "score": 0, "reason": "parse error — defaulted to fail"}
            for qid in batch_query_ids
        }


@register_tool('query_filter')
class DiverseQueryFilter(BaseTool):
    description = (
        "Filter a DiverseQuery JSON file, keeping only queries whose style and quality "
        "match the expected query type for the given task. Uses an LLM to make per-query "
        "pass/fail decisions using a 3-tier prompt: task context, query style guidelines, "
        "and golden query examples."
    )
    parameters = [
        {
            'name': 'input_path',
            'type': 'string',
            'description': 'Path to the DiverseQuery_Main_*.json file (Step 3 output).',
            'required': True,
        },
        {
            'name': 'task_name',
            'type': 'string',
            'description': 'Task name present in HighLevel_Task_Definition.',
            'required': True,
        },
        {
            'name': 'query_instruction',
            'type': 'string',
            'description': 'Loaded query style guide text (content of QueryInstrRefine_*.txt).',
            'required': False,
        },
        {
            'name': 'output_path',
            'type': 'string',
            'description': 'Path to write the filtered DiverseQuery JSON output.',
            'required': True,
        },
        {
            'name': 'batch_size',
            'type': 'integer',
            'description': 'Number of queries to send to the LLM in a single call.',
            'required': False,
        },
        {
            'name': 'inject_golden_queries',
            'type': 'boolean',
            'description': 'Whether to inject golden query examples into the filter prompt.',
            'required': False,
        },
        {
            'name': 'golden_query_count',
            'type': 'integer',
            'description': 'Maximum number of golden query examples to inject.',
            'required': False,
        },
        {
            'name': 'golden_query_max_chars',
            'type': 'integer',
            'description': 'Maximum characters kept for each injected golden query example.',
            'required': False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        params_dict = json5.loads(params)
        input_path = params_dict["input_path"]
        task_name = params_dict["task_name"]
        query_instruction = params_dict.get("query_instruction", "") or ""
        output_path = params_dict["output_path"]
        batch_size = int(params_dict.get("batch_size", 10))
        inject_golden_queries = bool(params_dict.get("inject_golden_queries", True))
        golden_query_count = max(0, int(params_dict.get("golden_query_count", 3)))
        golden_query_max_chars = max(50, int(params_dict.get("golden_query_max_chars", 200)))

        logger = kwargs.get("logger") or logging.getLogger(__name__)

        if task_name not in HighLevel_Task_Definition:
            return json5.dumps({"error": f"Task '{task_name}' not found in HighLevel_Task_Definition."})

        query_type, _doc_type, relevance_def = HighLevel_Task_Definition[task_name]

        llm_cfg = kwargs.get("llm_cfg", {
            "model": "gpt-4.1-mini",
            "model_server": os.getenv("OPENAI_BASE_URL"),
            "api_key": os.getenv("OPENAI_API_KEY"),
        })
        conc_enabled = bool(kwargs.get("concurrency_enabled", False))
        conc_max_workers = kwargs.get("max_workers", None)
        conc_jitter = float(kwargs.get("jitter", 0.0))

        # ---- Load DiverseQuery JSON ----
        with open(input_path, "r", encoding="utf-8") as f:
            dq_data = json.load(f)

        results = dq_data.get("results", [])

        # ---- Build golden query block ----
        golden_query_block = ""
        if inject_golden_queries:
            golden_query_block, injected_count = build_golden_query_block(
                task_name,
                max_examples=golden_query_count,
                max_chars=golden_query_max_chars,
            )
            if injected_count:
                logger.info(
                    "Query filter: injected %d golden query example(s) into prompt (max_chars=%d).",
                    injected_count, golden_query_max_chars,
                )
            else:
                logger.warning(
                    "Query filter: inject_golden_queries=true but no usable golden queries found for task=%s.",
                    task_name,
                )

        # ---- Flatten all queries into a single list ----
        # Each entry carries enough context to map back to the original result/query position.
        flat_queries: List[dict] = []
        for result_idx, result_item in enumerate(results):
            doc_id = result_item.get("doc_id", f"doc_{result_idx}")
            queries = result_item.get("queries", [])
            for query_idx, q_item in enumerate(queries):
                if isinstance(q_item, str):
                    query_text = q_item
                else:
                    query_text = q_item.get("query", "")
                if not query_text:
                    continue
                flat_queries.append({
                    "query_text": query_text,
                    "result_idx": result_idx,
                    "query_idx": query_idx,
                    "doc_id": doc_id,
                })

        total_queries = len(flat_queries)
        logger.info(
            "Query filter: task=%s query_type='%s' total_queries=%d batch_size=%d "
            "inject_golden=%s",
            task_name, query_type, total_queries, batch_size, inject_golden_queries,
        )

        if total_queries == 0:
            logger.warning("Query filter: no queries found in input; writing unchanged output.")
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dq_data, f, ensure_ascii=False, indent=2)
            return json5.dumps({
                "output_path": output_path,
                "total_queries": 0,
                "passed": 0,
                "filtered": 0,
                "score_distribution": {"score_3_perfect": 0, "score_2_good": 0, "score_1_borderline": 0, "score_0_mismatch": 0},
                "docs_with_all_filtered": 0,
            }, ensure_ascii=False)

        # ---- Split into batches of batch_size ----
        # Assign intra-batch query_ids q1, q2, ... for the LLM prompt.
        batches: List[List[dict]] = []
        for start in range(0, total_queries, batch_size):
            batches.append(flat_queries[start: start + batch_size])
        total_batches = len(batches)

        logger.info(
            "Query filter: %d batches (concurrency=%s, max_workers=%s, jitter=%.2f)",
            total_batches, conc_enabled, conc_max_workers, conc_jitter,
        )

        def _filter_one_batch(batch: List[dict], batch_num: int) -> dict:
            """Run a single LLM filter call and return verdict_map for the batch."""
            logger.info(
                "Query filter: batch %d/%d (queries %d-%d)",
                batch_num, total_batches,
                (batch_num - 1) * batch_size + 1,
                (batch_num - 1) * batch_size + len(batch),
            )
            # Assign intra-batch IDs: q1, q2, ...
            batch_items = [
                {"query_id": f"q{i + 1}", "query_text": entry["query_text"]}
                for i, entry in enumerate(batch)
            ]
            batch_ids = [item["query_id"] for item in batch_items]
            try:
                prompt = build_query_filter_prompt(
                    batch_items,
                    query_type,
                    relevance_def,
                    query_instruction,
                    golden_query_block=golden_query_block,
                )
                agent = Assistant(llm=llm_cfg)
                raw = ""
                for response in agent.run([{"role": "user", "content": prompt}]):
                    raw = response[-1]["content"]
                return parse_query_filter_response(raw, batch_ids, logger=logger)
            except Exception as exc:
                logger.error(
                    "Query filter: batch %d/%d failed (%s); defaulting all %d queries to 'fail'.",
                    batch_num, total_batches, exc, len(batch),
                )
                return {
                    f"q{i + 1}": {"verdict": "fail", "score": 0, "reason": "batch processing error"}
                    for i in range(len(batch))
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

        # ---- Collect global verdict by (result_idx, query_idx) ----
        # verdict_maps[b] is the map for batch b; intra-batch key is q1..qN
        global_verdicts: dict = {}  # (result_idx, query_idx) -> verdict info
        score_distribution = {0: 0, 1: 0, 2: 0, 3: 0}

        for batch, verdict_map in zip(batches, verdict_maps):
            for intra_idx, entry in enumerate(batch):
                qid = f"q{intra_idx + 1}"
                info = verdict_map.get(
                    qid,
                    {"verdict": "fail", "score": 0, "reason": "missing verdict — defaulted to fail"},
                )
                key = (entry["result_idx"], entry["query_idx"])
                global_verdicts[key] = info
                score = info.get("score", 0)
                if isinstance(score, int) and 0 <= score <= 3:
                    score_distribution[score] = score_distribution.get(score, 0) + 1

        # ---- Rebuild results keeping only passing queries ----
        passed_count = 0
        filtered_count = 0
        docs_with_all_filtered = 0

        new_results = []
        for result_idx, result_item in enumerate(results):
            queries = result_item.get("queries", [])
            passing_queries = []
            for query_idx, q_item in enumerate(queries):
                # Queries that had empty text were never sent to LLM; keep them as-is
                if isinstance(q_item, str):
                    query_text = q_item
                else:
                    query_text = q_item.get("query", "")
                if not query_text:
                    passing_queries.append(q_item)
                    continue
                key = (result_idx, query_idx)
                info = global_verdicts.get(
                    key,
                    {"verdict": "fail", "score": 0, "reason": "not evaluated"},
                )
                if info["verdict"] == "pass":
                    passed_count += 1
                    passing_queries.append(q_item)
                else:
                    filtered_count += 1
                    logger.debug(
                        "Query filter: FILTERED doc_id=%s query_idx=%d score=%s reason=%s",
                        result_item.get("doc_id", "?"), query_idx,
                        info.get("score", "?"), str(info.get("reason", ""))[:120],
                    )

            if not passing_queries:
                docs_with_all_filtered += 1
                logger.debug(
                    "Query filter: all queries filtered for doc_id=%s — removing from results.",
                    result_item.get("doc_id", "?"),
                )
                continue  # Drop result items that have no passing queries

            new_result = dict(result_item)
            new_result["queries"] = passing_queries
            new_results.append(new_result)

        # ---- Write filtered DQ JSON ----
        filter_stats = {
            "total_queries": total_queries,
            "passed": passed_count,
            "filtered": filtered_count,
            "docs_with_all_filtered": docs_with_all_filtered,
            "score_distribution": {
                "score_3_perfect": score_distribution.get(3, 0),
                "score_2_good": score_distribution.get(2, 0),
                "score_1_borderline": score_distribution.get(1, 0),
                "score_0_mismatch": score_distribution.get(0, 0),
            },
        }

        out_data = dict(dq_data)
        out_data["results"] = new_results
        out_data["_filter_stats"] = filter_stats

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        perfect = score_distribution.get(3, 0)
        good = score_distribution.get(2, 0)
        borderline = score_distribution.get(1, 0)
        mismatch = score_distribution.get(0, 0)

        logger.info(
            "Query filter done: total=%d passed=%d filtered=%d docs_all_filtered=%d → output=%s",
            total_queries, passed_count, filtered_count, docs_with_all_filtered, output_path,
        )
        logger.info(
            "Score distribution: perfect(3)=%d good(2)=%d borderline(1)=%d mismatch(0)=%d",
            perfect, good, borderline, mismatch,
        )

        return json5.dumps({
            "output_path": output_path,
            "total_queries": total_queries,
            "passed": passed_count,
            "filtered": filtered_count,
            "docs_with_all_filtered": docs_with_all_filtered,
            "score_distribution": {
                "score_3_perfect": perfect,
                "score_2_good": good,
                "score_1_borderline": borderline,
                "score_0_mismatch": mismatch,
            },
        }, ensure_ascii=False)


def filter_diverse_queries(config: QueryFilterConfig, logger=None) -> str:
    """Run query-style filtering on a DiverseQuery JSON file (Step 3 output).

    Mirrors the interface of ``filter_diverse_texts``: accepts a config object,
    returns the path of the filtered output file.

    Parameters
    ----------
    config : QueryFilterConfig
    logger : logging.Logger, optional

    Returns
    -------
    str
        Absolute path to the filtered DiverseQuery JSON file.
    """
    log = logger or logging.getLogger(__name__)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.output_dir, f"DiverseQuery_Filtered_{timestamp}.json")

    llm_cfg = {
        "model": config.model,
        "model_server": config.model_server or os.getenv("OPENAI_BASE_URL"),
        "api_key": config.api_key or os.getenv("OPENAI_API_KEY"),
    }

    tool = DiverseQueryFilter()
    result_json = tool.call(
        json5.dumps({
            "input_path": config.input_path,
            "task_name": config.task_name,
            "query_instruction": config.query_instruction,
            "output_path": output_path,
            "batch_size": config.batch_size,
            "inject_golden_queries": config.inject_golden_queries,
            "golden_query_count": config.golden_query_count,
            "golden_query_max_chars": config.golden_query_max_chars,
        }),
        llm_cfg=llm_cfg,
        logger=log,
        concurrency_enabled=config.concurrency_enabled,
        max_workers=config.max_workers,
        jitter=config.jitter,
    )

    result = json5.loads(result_json)
    if "error" in result:
        raise RuntimeError(f"Query filter failed: {result['error']}")

    log.info(
        "=== Step 3.5 done === output=%s total=%d passed=%d filtered=%d docs_all_filtered=%d",
        result["output_path"],
        result["total_queries"],
        result["passed"],
        result["filtered"],
        result["docs_with_all_filtered"],
    )
    return result["output_path"]


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _parse_cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter a Step-3 DiverseQuery JSON with an LLM (style + golden-query benchmark).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Query_Filter.py -t touche2020 -i ./DiverseQuery_Main_20260101_120000.json

  python Query_Filter.py -t touche2020 -i ./dq.json \\
      --query-instruction-file ./QueryInstrRefine_touche2020_20260423_001330.txt

  python Query_Filter.py -t msmarco -i ./dq.json -o ./out_dir --batch-size 8

Environment (defaults if flags omitted):
  OPENAI_MODEL, OPENAI_BASE_URL, OPENAI_API_KEY
""",
    )
    p.add_argument(
        "-t", "--task",
        required=True,
        help="Task id (must exist in HighLevel_Task_Definition), e.g. touche2020, msmarco",
    )
    p.add_argument(
        "-i", "--input",
        required=True,
        dest="input_path",
        help="Path to DiverseQuery JSON (same schema as Step 3 output: results[].queries[])",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--query-instruction",
        default="",
        metavar="TEXT",
        help="Inline query style guide text (QueryInstrRefine-style). Empty = omit style tier.",
    )
    g.add_argument(
        "--query-instruction-file",
        metavar="PATH",
        dest="query_instruction_file",
        help="Read query style guide from this UTF-8 text file (overrides --query-instruction).",
    )
    p.add_argument(
        "-o", "--output-dir",
        default="Results",
        help="Directory for DiverseQuery_Filtered_<timestamp>.json (default: Results)",
    )
    p.add_argument(
        "--log-dir",
        default="Results/Logs",
        help="Directory for Query_Filter_*.log when using module default logging (default: Results/Logs)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Queries per LLM call (default: 10)",
    )
    p.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="LLM model name (default: env OPENAI_MODEL or gpt-4o-mini)",
    )
    p.add_argument(
        "--model-server",
        default=os.getenv("OPENAI_BASE_URL"),
        help="OpenAI-compatible base URL (default: env OPENAI_BASE_URL)",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key (default: env OPENAI_API_KEY)",
    )
    p.add_argument(
        "--inject-golden-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject Few_Shot_Example queries into the prompt (default: true)",
    )
    p.add_argument(
        "--golden-query-count",
        type=int,
        default=3,
        help="Max golden query examples to inject (default: 3)",
    )
    p.add_argument(
        "--golden-query-max-chars",
        type=int,
        default=200,
        help="Max chars per golden query snippet (default: 200)",
    )
    p.add_argument(
        "--concurrency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable concurrent LLM batches (default: false)",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max worker threads when --concurrency (default: library default)",
    )
    p.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Random delay (seconds) before each batch submit when --concurrency (default: 0)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()

    query_instr = args.query_instruction or ""
    if getattr(args, "query_instruction_file", None):
        path = args.query_instruction_file
        with open(path, "r", encoding="utf-8") as f:
            query_instr = f.read()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    cfg = QueryFilterConfig(
        input_path=args.input_path,
        task_name=args.task.strip().lower(),
        query_instruction=query_instr,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        batch_size=max(1, args.batch_size),
        model=args.model,
        model_server=args.model_server,
        api_key=args.api_key,
        concurrency_enabled=args.concurrency,
        max_workers=args.max_workers,
        jitter=args.jitter,
        inject_golden_queries=args.inject_golden_queries,
        golden_query_count=max(0, args.golden_query_count),
        golden_query_max_chars=max(50, args.golden_query_max_chars),
    )

    out = filter_diverse_queries(cfg)
    logging.info("Filtered output: %s", out)
