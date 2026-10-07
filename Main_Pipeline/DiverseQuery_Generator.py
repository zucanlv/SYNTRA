"""
Generate diverse queries from IdAttr-style JSONL (each line: doc_id, title, text, identifiers, attributes).

Usage:
  conda activate DSA
  python DiverseQuery_Generator.py --input /path/to/IdAttr_xxx.jsonl

Example:
  python DiverseQuery_Generator.py --input Results/IdAttr_20260304_155624.jsonl --num_queries 5 --model gpt-4o-mini
"""

import os
import json
import json5
import logging
from typing import Any, Dict, List, Optional, Tuple
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from datetime import datetime
from Few_Shot_Formatter import format_examples, get_examples
from HighLevel_Def import HighLevel_Task_Definition

# Ensure log/output dirs exist before FileHandler is created
# The pipeline entry point owns logging when this module is imported.
if __name__ == "__main__":
    os.makedirs("Logs", exist_ok=True)
    os.makedirs("Results", exist_ok=True)

    # Configure logging to write to both a file and the console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"Logs/DiverseQuery_Generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

def _safe_json5_loads(text: str) -> Dict[str, Any]:
    try:
        loaded = json5.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    return {}


def _normalize_identifiers(identifiers: Any) -> List[Dict[str, Any]]:
    if not isinstance(identifiers, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in identifiers:
        if isinstance(it, dict) and isinstance(it.get("identifier"), str):
            out.append(it)
        elif isinstance(it, str):
            out.append({"type": "entity", "identifier": it, "label": it[:48]})
    # de-dup by identifier string
    seen = set()
    deduped = []
    for it in out:
        key = it.get("identifier")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped


def _normalize_attributes(attributes: Any) -> List[Dict[str, Any]]:
    if not isinstance(attributes, list):
        return []
    out: List[Dict[str, Any]] = []
    for dim in attributes:
        if not isinstance(dim, dict):
            continue
        name = dim.get("dimension")
        options = dim.get("options")
        if not isinstance(name, str) or not isinstance(options, list):
            continue
        options_norm = [o for o in options if isinstance(o, str)]
        out.append({"dimension": name, "options": options_norm})
    # de-dup by dimension
    seen = set()
    deduped = []
    for dim in out:
        key = dim["dimension"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dim)
    return deduped


def _parse_input_json(path: str) -> List[Dict[str, Any]]:
    """
    Parse JSONL: each line is {doc_id, title, text, identifiers, attributes}.
    Returns a list of items preserving doc_id and title alongside text/identifiers/attributes.
    """
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _safe_json5_loads(line)
            if not obj:
                try:
                    obj = json.loads(line)
                except Exception as e:
                    logging.warning(f"Failed to parse line: {e}")
                    continue
            if not isinstance(obj, dict):
                continue
            doc = obj.get("doc")
            if not isinstance(doc, str) or not doc.strip():
                continue
            items.append({
                "doc_id": obj.get("doc_id", ""),
                "doc": doc.strip(),
                "identifiers": _normalize_identifiers(obj.get("identifiers", [])),
                "attributes": _normalize_attributes(obj.get("attributes", [])),
            })
    return items


def _validate_queries(
    queries: Any,
    identifiers: List[Dict[str, Any]],
    attributes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Keeps only well-formed query items:
    - query: non-empty string
    - used_identifier: exactly 1 item; must be present in provided identifiers (by identifier string)
    - used_attributes: one option per dimension (subset allowed but must be from provided options)
    """
    if not isinstance(queries, list):
        return []

    valid_ident_set = {i.get("identifier") for i in identifiers if isinstance(i, dict)}
    attr_dim_to_opts = {
        a["dimension"]: set(a.get("options", []))
        for a in attributes
        if isinstance(a, dict) and isinstance(a.get("dimension"), str)
    }

    out: List[Dict[str, Any]] = []
    for q in queries:
        if not isinstance(q, dict):
            continue
        text = q.get("query")
        if not isinstance(text, str) or not text.strip():
            continue
        used_ids = q.get("used_identifier", [])
        if not isinstance(used_ids, list):
            continue
        used_ids_norm = [x for x in used_ids if isinstance(x, str) and x in valid_ident_set]
        used_attrs = q.get("used_attributes", {})
        if not isinstance(used_attrs, dict):
            continue
        # If any dim or opt is not in our preset, this query is invalid; discard it.
        used_attrs_norm: Dict[str, str] = {}
        invalid_attrs = False
        for dim, opt in used_attrs.items():
            if not isinstance(dim, str) or not isinstance(opt, str):
                invalid_attrs = True
                break
            if dim not in attr_dim_to_opts or opt not in attr_dim_to_opts[dim]:
                invalid_attrs = True
                break
            used_attrs_norm[dim] = opt
        if invalid_attrs or not used_attrs_norm:
            continue
        out.append({
            "query": text.strip(),
            "used_identifier": used_ids_norm,
            "used_attributes": used_attrs_norm,
        })
    return out


def _canon_plan_key(used_identifier: List[str], used_attributes: Dict[str, str]) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
    """
    Canonical key for de-duplication/validation.
    Identifiers are treated as an unordered set; attributes are sorted by dimension.
    """
    ids_key = tuple(sorted([x for x in used_identifier if isinstance(x, str)]))
    attrs_key = tuple(sorted([(k, v) for k, v in used_attributes.items() if isinstance(k, str) and isinstance(v, str)]))
    return ids_key, attrs_key


def _validate_plans(
    plans: Any,
    identifiers: List[Dict[str, Any]],
    attributes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Keeps only well-formed plan items:
    - used_identifier: exactly 1 item; must be present in provided identifiers (by identifier string)
    - used_attributes: MUST include exactly one option per provided dimension (no missing dims)
    """
    if not isinstance(plans, list):
        return []

    valid_ident_set = {i.get("identifier") for i in identifiers if isinstance(i, dict)}
    attr_dim_to_opts = {
        a["dimension"]: set(a.get("options", []))
        for a in attributes
        if isinstance(a, dict) and isinstance(a.get("dimension"), str)
    }
    required_dims = list(attr_dim_to_opts.keys())

    out: List[Dict[str, Any]] = []
    for p in plans:
        if not isinstance(p, dict):
            continue
        used_ids = p.get("used_identifier", [])
        if isinstance(used_ids, str):
            used_ids = [used_ids]
        elif not isinstance(used_ids, list):
            continue
        used_ids_norm = [x for x in used_ids if isinstance(x, str) and x in valid_ident_set]
        if len(used_ids_norm) != 1:
            continue

        used_attrs = p.get("used_attributes", {})
        if not isinstance(used_attrs, dict):
            continue
        # Must cover all required dimensions and only valid options.
        if set(used_attrs.keys()) != set(required_dims):
            continue
        invalid = False
        used_attrs_norm: Dict[str, str] = {}
        for dim in required_dims:
            opt = used_attrs.get(dim)
            if not isinstance(opt, str) or opt not in attr_dim_to_opts.get(dim, set()):
                invalid = True
                break
            used_attrs_norm[dim] = opt
        if invalid:
            continue

        out.append({"used_identifier": used_ids_norm, "used_attributes": used_attrs_norm})
    return out


def _validate_queries_against_plans(
    queries: Any,
    plans: List[Dict[str, Any]],
    identifiers: List[Dict[str, Any]],
    attributes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Validate queries AND ensure each query matches an existing plan (same identifier set + same attributes).
    Returns queries normalized to use plan's used_identifier/used_attributes exactly.
    """
    plans_valid = _validate_plans(plans, identifiers=identifiers, attributes=attributes)
    plan_map: Dict[Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}
    for p in plans_valid:
        plan_map[_canon_plan_key(p["used_identifier"], p["used_attributes"])] = p

    base_valid = _validate_queries(queries, identifiers=identifiers, attributes=attributes)
    out: List[Dict[str, Any]] = []
    for q in base_valid:
        key = _canon_plan_key(q.get("used_identifier", []), q.get("used_attributes", {}))
        plan = plan_map.get(key)
        if not plan:
            continue
        out.append({
            "query": q["query"],
            # Normalize to the plan values so downstream is consistent.
            "used_identifier": plan["used_identifier"],
            "used_attributes": plan["used_attributes"],
        })
    return out


def build_plan_prompt(
    doc: str,
    identifiers: List[Dict[str, Any]],
    attributes: List[Dict[str, Any]],
    num_plans: int,
    already_selected: Optional[List[Dict[str, Any]]] = None,
    task_instruction: Optional[str] = None,
    task_name: Optional[str] = None,
) -> str:
    """
    Stage 1: only plan (identifier + attribute group), no query text.
    """
    identifiers_brief = [
        {
            "type": it.get("type", "entity"),
            "identifier": it.get("identifier", ""),
            "label": it.get("label", ""),
        }
        for it in identifiers
        if isinstance(it, dict)
    ]
    already_selected = already_selected or []
    task_instruction = task_instruction or ""
    task_name = task_name or ""
    query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]

    few_shot_section = format_examples(get_examples(task_name))

    return f"""
You are planning diverse search query "blueprints" for information retrieval.
**You will be given:**
1) A {doc_type} (the ONLY source of truth)
2) A set of identifiers (information anchors extracted from the document)
3) A set of query attribute dimensions (each has multiple options)

**Task Context:**
- Target Output (Query Type): {query_type}
- Source Input (Document Type): {doc_type}
- Relevance Criteria: {relevance_def}

**Few-Shot Examples:**
{few_shot_section}

**Your task:** generate exactly {num_plans} diverse BLUEPRINTS. Each blueprint contains:
- exactly 1 chosen identifier (copy EXACTLY from the provided identifiers list)
- ONE attribute group: choose exactly ONE option for EACH attribute dimension

**Important:** In this stage, DO NOT write any search queries. Only output the selections (identifiers + attributes).

**Task-Specific Style Guidance:**
{task_instruction}

Use this guidance to inform your blueprint planning decisions.

**Hard constraints:**
- Each blueprint must contain exactly 1 identifier — no more, no less.
- Do NOT invent identifiers or attribute options.
- Avoid redundancy: do not repeat the same identifier+attribute-group combination.

**Diversity planning (across these {num_plans} blueprints, and considering any already selected blueprints):**
- Maximize coverage of different identifiers.
- For each attribute dimension, try to cover as many different options as possible.

**Already selected blueprints (DO NOT repeat these combinations):**
{json.dumps(already_selected, ensure_ascii=False, indent=2)}

**Input document:**
{doc}

**Identifiers (choose from these; copy identifiers exactly):**
{json.dumps(identifiers_brief, ensure_ascii=False, indent=2)}

**Attribute dimensions (choose exactly one option per dimension per blueprint):**
{json.dumps(attributes, ensure_ascii=False, indent=2)}

**Output format requirements (MANDATORY):**
- Output MUST be a single valid JSON object (RFC 8259).
- Output MUST contain ONLY the JSON object: no Markdown, no code fences, no commentary, no extra keys, no trailing text.
- Do NOT include any "already_selected" plans in your output (do not repeat them); output ONLY NEW plans in the "plans" object.
- Use double quotes for all keys and strings. No trailing commas.
- JSON must start with '{{' and end with '}}'.

**JSON schema (fixed keys; output MUST follow exactly):**
{{
  "plans": [
    {{
      "used_identifier": ["identifier string"],
      "used_attributes": {{
        "dimension name": "one option",
        "dimension name": "one option"
      }}
    }}
  ]
}}
""".strip()


def build_query_prompt_from_plans(
    doc: str,
    identifiers: List[Dict[str, Any]],
    attributes: List[Dict[str, Any]],
    plans: List[Dict[str, Any]],
    task_instruction: Optional[str] = None,
    task_name: Optional[str] = None,
) -> str:
    """
    Stage 2: write queries for a given set of plans (no reselection allowed).
    """
    identifiers_brief = [
        {
            "type": it.get("type", "entity"),
            "identifier": it.get("identifier", ""),
            "label": it.get("label", ""),
        }
        for it in identifiers
        if isinstance(it, dict)
    ]
    task_instruction = task_instruction or ""
    task_name = task_name or ""
    query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]

    few_shot_section = format_examples(get_examples(task_name))

    return f"""
You are a world-class synthetic query writer for information retrieval.

**You will be given:**
1) A {doc_type} (the ONLY source of truth)
2) A set of identifiers (information anchors extracted from the document)
3) A set of query attribute dimensions (each has multiple options)
4) A list of BLUEPRINTS (each blueprint already specifies exactly 1 identifier and exactly one option per attribute dimension)

**Task Context:**
- Target Output (Query Type): {query_type}
- Source Input (Document Type): {doc_type}
- Relevance Criteria: {relevance_def}

**Few-Shot Examples:**
{few_shot_section}

**Your task:** Generate a list of {len(plans)} distinct queries, strictly adhering to the provided blueprints.

**Task-Specific Style Guidance:**
{task_instruction}

**Task Execution:**
1. Iterate through the provided Blueprints one by one.
2. Apply the Task-Specific Style Guidance to determine the appropriate linguistic register and domain-specific terminology (e.g., professional, academic, or casual).
3. For each blueprint, craft a high-quality query that implicitly but clearly reflects all assigned Attributes and targets the specified Identifiers.
4. Ensure the query is a perfect marriage of the blueprint's rigid requirements and the domain's stylistic nuances.

**NO reselection rule (STRICT):**
- You MUST NOT change the blueprint selections.
- For each query item in the output, you MUST include the blueprint's used_identifier and used_attributes EXACTLY as separate JSON fields (for reference). Do NOT embed them inside the "query" text.

**Attribute expressibility rule (STRICT):**
- The query text must clearly reflect the chosen options for each dimension (observable from wording).
- Even if a blueprint presents a logical challenge, you must provide the most plausible grounded query without altering the constraints.

**Input document:**
{doc}

**Identifiers reference:**
{json.dumps(identifiers_brief, ensure_ascii=False, indent=2)}

**Attribute dimensions reference:**
{json.dumps(attributes, ensure_ascii=False, indent=2)}

**Blueprints (write one query per blueprint, in the same order):**
{json.dumps(plans, ensure_ascii=False, indent=2)}

**Output format requirements (MANDATORY):**
- Output MUST be a single valid JSON object (RFC 8259).
- Output MUST contain ONLY the JSON object: no Markdown, no code fences, no commentary, no extra keys, no trailing text.
- Use double quotes for all keys and strings. No trailing commas.
- JSON must start with '{{' and end with '}}'.

**JSON schema (fixed keys; output MUST follow exactly):**
{{
  "queries": [
    {{
      "query": "string (one query matching the task-specified query type)",
      "used_identifier": ["identifier string"],
      "used_attributes": {{
        "dimension name": "one option",
        "dimension name": "one option"
      }}
    }}
  ]
}}
""".strip()


def build_doc_only_query_prompt(
    doc: str,
    num_queries: int,
    task_instruction: Optional[str] = None,
    task_name: Optional[str] = None,
) -> str:
    """Build the direct query generation prompt used when IdAttr is disabled."""
    task_instruction = task_instruction or ""
    task_name = task_name or ""
    query_type, doc_type, relevance_def = HighLevel_Task_Definition[task_name]
    few_shot_section = format_examples(get_examples(task_name))

    return f"""
You are a world-class synthetic query writer for information retrieval.

**You will be given:**
1) A {doc_type} (the ONLY source of truth)

**Task Context:**
- Target Output (Query Type): {query_type}
- Source Input (Document Type): {doc_type}
- Relevance Criteria: {relevance_def}

**Few-Shot Examples:**
{few_shot_section}

**Your task:** Generate exactly {num_queries} distinct queries.

**Task-Specific Style Guidance:**
{task_instruction}

**Input document:**
{doc}

**Output format requirements (MANDATORY):**
- Output MUST be a single valid JSON object (RFC 8259).
- Output MUST contain ONLY the JSON object: no Markdown, no code fences, no commentary, no extra keys, no trailing text.
- Use double quotes for all keys and strings. No trailing commas.
- JSON must start with '{{' and end with '}}'.

**JSON schema (fixed keys; output MUST follow exactly):**
{{
  "queries": [
    {{
      "query": "string (one query matching the task-specified query type)"
    }}
  ]
}}
""".strip()


def _validate_doc_only_queries(queries: Any) -> List[Dict[str, Any]]:
    """Normalize doc-only query outputs to the same downstream query object schema."""
    if not isinstance(queries, list):
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for q in queries:
        if isinstance(q, str):
            text = q
        elif isinstance(q, dict):
            text = q.get("query")
        else:
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        if text in seen:
            continue
        seen.add(text)
        out.append({
            "query": text,
            "used_identifier": [],
            "used_attributes": {},
        })
    return out


@register_tool('diverse_query_generator')
class DiverseQueryGenerator(BaseTool):
    description = "Generate diverse queries for a given document, based on the identifiers and attributes generated by the IdentifierAttributeGenerator tool."
    parameters = [
    {
        'name': 'doc',
        'type': 'string',
        'description': 'The document (title + text combined) to generate diverse queries for.',
        'required': True
    },
    {
        'name': 'identifiers',
        'type': 'list',
        'description': 'The identifiers generated by the IdentifierAttributeGenerator tool.',
        'required': False
    },
    {
        'name': 'attributes',
        'type': 'list',
        'description': 'The attributes generated by the IdentifierAttributeGenerator tool.',
        'required': False
    },
    {
        'name': 'num_queries',
        'type': 'number',
        'description': 'How many diverse queries to generate for this document.',
        'required': False
    },
    {
        'name': 'task_name',
        'type': 'string',
        'description': 'Task name (e.g. "msmarco", "arguana", "trec-covid") from the high-level definitions.',
        'required': True
    },
    {
        'name': 'task_instruction',
        'type': 'string',
        'description': 'Task-specific style guidance to steer query generation.',
        'required': False
    },
    {
        'name': 'use_idattr',
        'type': 'boolean',
        'description': 'Whether to use identifier/attribute blueprint planning. Set false for doc-only generation.',
        'required': False
    },
    ]

    def call(self, params: str, **kwargs) -> str:
        params_dict = json5.loads(params)
        doc = params_dict.get('doc')
        identifiers = _normalize_identifiers(params_dict.get('identifiers'))
        attributes = _normalize_attributes(params_dict.get('attributes'))
        num_queries = params_dict.get('num_queries', 8)
        task_name = params_dict.get('task_name', '')
        task_instruction = params_dict.get('task_instruction', '')
        use_idattr = bool(params_dict.get('use_idattr', True))

        if task_name not in HighLevel_Task_Definition:
            return json5.dumps({'error': f"Task '{task_name}' not found."}, ensure_ascii=False)

        try:
            num_queries = int(num_queries)
        except Exception:
            num_queries = 8
        num_queries = max(1, num_queries)

        if not isinstance(doc, str) or not doc.strip():
            return json5.dumps({'error': 'doc must be a non-empty string'}, ensure_ascii=False)
        if use_idattr and not identifiers:
            return json5.dumps({'error': 'identifiers must be a non-empty list'}, ensure_ascii=False)
        if use_idattr and not attributes:
            return json5.dumps({'error': 'attributes must be a non-empty list'}, ensure_ascii=False)

        # Internal agent to generate outputs
        llm_cfg = kwargs.get('llm_cfg', {
            'model': 'gpt-4o-mini',
            'model_server': os.getenv('OPENAI_BASE_URL'),
            'api_key': os.getenv('OPENAI_API_KEY'),
        })

        designer = Assistant(llm=llm_cfg)
        raw_plan_output = ""
        raw_query_outputs: List[str] = []

        if not use_idattr:
            query_prompt = build_doc_only_query_prompt(
                doc=doc,
                num_queries=num_queries,
                task_instruction=task_instruction,
                task_name=task_name,
            )
            tmp_raw = ""
            for response in designer.run([{'role': 'user', 'content': query_prompt}]):
                tmp_raw = response[-1]['content']
            raw_query_outputs.append(tmp_raw)

            try:
                parsed_q = json5.loads(tmp_raw)
            except Exception as e:
                logging.warning(f"Failed to parse doc-only query JSON output: {e}")
                parsed_q = {}

            payload = {
                "doc": doc,
                "identifiers": [],
                "attributes": [],
                "queries": _validate_doc_only_queries(parsed_q.get("queries", []))[:num_queries],
                "raw_plan_output": "",
                "raw_query_output": "\n\n".join(raw_query_outputs).strip(),
                "raw_model_output": tmp_raw,
                "use_idattr": False,
            }
            return json5.dumps(payload, ensure_ascii=False)

        # -------------------------
        # Stage 1: plan blueprints
        # -------------------------
        plans: List[Dict[str, Any]] = []
        seen_plan_keys = set()
        max_plan_rounds = 3
        for _round in range(max_plan_rounds):
            remaining = num_queries - len(plans)
            if remaining <= 0:
                break

            plan_prompt = build_plan_prompt(
                doc=doc,
                identifiers=identifiers,
                attributes=attributes,
                num_plans=remaining,
                already_selected=plans,
                task_instruction=task_instruction,
                task_name=task_name,
            )
            tmp_raw = ""
            for response in designer.run([{'role': 'user', 'content': plan_prompt}]):
                tmp_raw = response[-1]['content']
            # Preserve the first round raw output (most informative); append subsequent rounds.
            raw_plan_output = (raw_plan_output + "\n\n" + tmp_raw).strip() if raw_plan_output else tmp_raw

            try:
                parsed_plan = json5.loads(tmp_raw)
            except Exception as e:
                logging.warning(f"Failed to parse plan JSON output: {e}")
                parsed_plan = {}

            new_plans = _validate_plans(parsed_plan.get("plans", []), identifiers=identifiers, attributes=attributes)
            for p in new_plans:
                if len(plans) >= num_queries:
                    break
                key = _canon_plan_key(p["used_identifier"], p["used_attributes"])
                if key in seen_plan_keys:
                    continue
                seen_plan_keys.add(key)
                plans.append(p)

        if len(plans) < num_queries:
            logging.warning(f"Only planned {len(plans)}/{num_queries} blueprints; proceeding with available plans.")

        # -------------------------
        # Stage 2: write queries
        # -------------------------
        queries: List[Dict[str, Any]] = []
        # Chunking reduces attention overload when num_queries is large.
        chunk_size = 12
        remaining_plans = plans[:]
        max_query_rounds = 2
        for _round in range(max_query_rounds):
            if not remaining_plans:
                break
            generated_this_round: List[Dict[str, Any]] = []

            for i in range(0, len(remaining_plans), chunk_size):
                chunk_plans = remaining_plans[i:i + chunk_size]
                query_prompt = build_query_prompt_from_plans(
                    doc=doc,
                    identifiers=identifiers,
                    attributes=attributes,
                    plans=chunk_plans,
                    task_instruction=task_instruction,
                    task_name=task_name,
                )
                tmp_raw = ""
                for response in designer.run([{'role': 'user', 'content': query_prompt}]):
                    tmp_raw = response[-1]['content']
                raw_query_outputs.append(tmp_raw)

                try:
                    parsed_q = json5.loads(tmp_raw)
                except Exception as e:
                    logging.warning(f"Failed to parse query JSON output: {e}")
                    parsed_q = {}

                chunk_queries = _validate_queries_against_plans(
                    parsed_q.get("queries", []),
                    plans=chunk_plans,
                    identifiers=identifiers,
                    attributes=attributes,
                )
                generated_this_round.extend(chunk_queries)

            # De-dup by plan key (one query per plan).
            existing_keys = {_canon_plan_key(q["used_identifier"], q["used_attributes"]) for q in queries}
            for q in generated_this_round:
                if len(queries) >= num_queries:
                    break
                key = _canon_plan_key(q["used_identifier"], q["used_attributes"])
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                queries.append(q)

            # Determine which plans still need queries.
            query_keys = {_canon_plan_key(q["used_identifier"], q["used_attributes"]) for q in queries}
            remaining_plans = [p for p in plans if _canon_plan_key(p["used_identifier"], p["used_attributes"]) not in query_keys]
            if len(queries) >= num_queries:
                remaining_plans = []

        if len(queries) < len(plans):
            logging.warning(f"Only generated {len(queries)}/{len(plans)} queries matching planned blueprints.")

        payload = {
            "doc": doc,
            "identifiers": identifiers,
            "attributes": attributes,
            "queries": queries[:num_queries],
            "raw_plan_output": raw_plan_output,
            "raw_query_output": "\n\n".join(raw_query_outputs).strip(),
            "raw_model_output": (raw_query_outputs[-1] if raw_query_outputs else ""),
        }
        return json5.dumps(payload, ensure_ascii=False)

if __name__ == "__main__":
    import argparse

    llm_cfg = {
        'model': 'gpt-4o-mini',
        'model_server': os.getenv('OPENAI_BASE_URL'),
        'api_key': os.getenv('OPENAI_API_KEY'),
    }

    valid_task_names = list(HighLevel_Task_Definition.keys())

    parser = argparse.ArgumentParser(description="Generate diverse queries per document.")
    parser.add_argument("--input", type=str, default="", help="Input JSONL path (each line: doc_id, doc, identifiers, attributes).")
    parser.add_argument("--num_queries", type=int, default=24, help="Queries per document.")
    parser.add_argument("--task_name", type=str, default="", help=f"Task name, one of: {valid_task_names}.")
    parser.add_argument("--task_instruction", type=str, default="", help="Task-specific style guidance string (inline text or path to a .txt file).")
    parser.add_argument("--model", type=str, default=llm_cfg["model"], help="Model name.")
    parser.add_argument("--model_server", type=str, default=llm_cfg["model_server"], help="OpenAI-compatible base URL.")
    parser.add_argument("--api_key", type=str, default=llm_cfg["api_key"] or "", help="API key (overrides OPENAI_API_KEY).")
    args = parser.parse_args()

    # --task_instruction can be a file path or an inline string
    task_instruction: str = args.task_instruction or ""
    if task_instruction and os.path.isfile(task_instruction):
        with open(task_instruction, "r", encoding="utf-8") as _f:
            task_instruction = _f.read().strip()

    task_name: str = args.task_name or ""
    if args.input and not task_name:
        raise SystemExit("--task_name is required when --input is provided. "
                         f"Valid values: {valid_task_names}")
    if task_name and task_name not in HighLevel_Task_Definition:
        raise SystemExit(f"Unknown --task_name {task_name!r}. Valid values: {valid_task_names}")

    if args.model:
        llm_cfg["model"] = args.model
    if args.model_server:
        llm_cfg["model_server"] = args.model_server
    if args.api_key:
        llm_cfg["api_key"] = args.api_key

    tool = DiverseQueryGenerator()

    if args.input:
        items = _parse_input_json(args.input)
        if not items:
            raise SystemExit(f"Failed to parse input or no valid documents: {args.input}")

        results: List[Dict[str, Any]] = []
        for idx, it in enumerate(items):
            logging.info(f"Generating queries for document {idx + 1}/{len(items)} (doc_id={it.get('doc_id', '')!r})...")
            res_text = tool.call(
                json5.dumps(
                    {
                        "doc": it["doc"],
                        "identifiers": it["identifiers"],
                        "attributes": it["attributes"],
                        "num_queries": args.num_queries,
                        "task_name": task_name,
                        "task_instruction": task_instruction,
                    },
                    ensure_ascii=False,
                ),
                llm_cfg=llm_cfg,
            )
            result = _safe_json5_loads(res_text) or {"raw": res_text}
            result["doc_id"] = it.get("doc_id", "")
            results.append(result)

        out_path = f"Results/DiverseQuery_Batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_path": args.input,
                    "num_queries": args.num_queries,
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logging.info(f"Batch results saved to: {out_path}")

        # 额外输出：仅包含所有 query，每行一条（JSONL，便于后续 json.loads 逐行读取）
        all_queries: List[str] = []
        for r in results:
            if isinstance(r, dict) and "queries" in r:
                for q in r["queries"]:
                    if isinstance(q, dict) and isinstance(q.get("query"), str):
                        all_queries.append(q["query"].strip())
        queries_path = out_path.replace("DiverseQuery_Batch_", "DiverseQuery_Queries_").replace(".json", ".jsonl")
        with open(queries_path, "w", encoding="utf-8") as f:
            for q in all_queries:
                f.write(json.dumps({"query": q}, ensure_ascii=False) + "\n")
        logging.info(f"Queries-only list saved to: {queries_path} ({len(all_queries)} queries)")
    else:
        # Minimal local smoke test (no file): uses the same example passage as IdAttr_Generator.
        sample_passage = "The Houston Astros hope to make their first appearance in the American League Division Series, but they first will have to beat the New York Yankees in Tuesday night’s AL Wild Card Game. The game will be played at Yankee Stadium because New York finished with the better regular-season record. The pitching matchup is expected to be Masahiro Tanaka (Yankees) versus Dallas Keuchel (Astros). The winner will travel West to play the Kansas City Royals, who finished with the AL’s best record at 95-67."
        identifiers = [
            {"type": "entity", "identifier": "Houston Astros", "label": "Team: Houston Astros"},
            {"type": "entity", "identifier": "New York Yankees", "label": "Team: New York Yankees"},
            {"type": "entity", "identifier": "Yankee Stadium", "label": "Location: Yankee Stadium"},
            {"type": "entity", "identifier": "Masahiro Tanaka", "label": "Player: Masahiro Tanaka"},
        ]
        attributes = [
            {"dimension": "query intent", "options": ["who", "what", "when", "why"]},
            {"dimension": "answer form", "options": ["name", "date", "list", "number"]},
            {"dimension": "temporal focus", "options": ["before", "during", "after"]},
        ]
        result = tool.call(
            json5.dumps(
                {
                    "doc": sample_passage,
                    "identifiers": identifiers,
                    "attributes": attributes,
                    "num_queries": args.num_queries,
                    "task_name": task_name or "msmarco",
                    "task_instruction": task_instruction,
                },
                ensure_ascii=False,
            ),
            llm_cfg=llm_cfg,
        )
        logging.info(f"Tool output:\n{result}")
