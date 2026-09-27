"""
Hard Negative Mining Tool — Qwen-Agent BaseTool wrapper.

Works entirely on pre-retrieved Query2Passage output (same input format as
BatchCandidateAnnotator) and applies the Branch A/B KNN negative heuristic
from the reference implementation.

Branch A (dense region): when the 200th candidate still scores ≥ 0.95× the
  query–positive similarity, sample hard negatives from the mid-rank window
  [range_start : range_end] (default 30:200) to avoid trivial top-1 matches.
Branch B (sparse region): walk the ranked list from the top and collect
  candidates whose score < 0.95× the query–positive similarity.

When origin_doc_annotation=True: heuristic mining runs first for every query
(equivalent to origin_doc_annotation=False). Then the LLM (AnnoInstrRefine_*)
verifies only mined rows; rows whose original doc is not classified positive
are dropped. Two JSONL files are written — full mined set and origin-filtered
set — so hard negatives stay aligned between modes within one run.

When origin_doc_annotation=False: single JSONL output (full mined only).

Output JSONL: one JSON object per source document with a ``queries`` list;
each query object includes pos/neg texts and doc_ids.
"""

import os
import json
import json5
import logging
import random
import argparse
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from datetime import datetime
from tqdm import tqdm

from concurrent_runner import run_concurrent

import ijson

from q2p_json_stream import extract_input_path_prefix, file_is_large_q2p


def _score_as_float(v: Any) -> float:
    """Coerce retrieval scores to float (ijson parses JSON numbers as Decimal)."""
    if v is None:
        return 0.0
    if isinstance(v, Decimal):
        return float(v)
    return float(v)


INSTRUCTIONS_DIR = Path(__file__).resolve().parent / "Results" / "Instructions"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@register_tool("hn_mine")
class HnMineTool(BaseTool):
    """Mine hard negatives from Query2Passage output using KNN density heuristics.

    Accepts the same input_path / output_path interface as BatchCandidateAnnotator
    so it can be dropped in as an alternative Step 5 without changing caller code.
    """

    description = (
        "Mine hard negatives for each query from pre-retrieved Query2Passage candidates. "
        "Applies Branch A (mid-rank sampling in dense regions) or Branch B (score-threshold "
        "selection in sparse regions). When origin_doc_annotation is true, mines first then "
        "LLM-filters by original-doc relevance; writes both full-mined and filtered JSONL."
    )
    parameters = [
        {"name": "input_path", "type": "string",
         "description": "Path to Query2Passage JSON output file.", "required": True},
        {"name": "output_path", "type": "string",
         "description": "Output JSONL path for full mined results (same as origin_doc_annotation=false).",
         "required": False},
        {"name": "output_path_origin_filtered", "type": "string",
         "description": "When origin_doc_annotation=true: JSONL path for LLM-filtered rows; "
         "if omitted, derived from output_path as <stem>_origin_pos.jsonl.",
         "required": False},
        {"name": "task_name", "type": "string",
         "description": "Task name (e.g. msmarco). Used to load annotation instruction.",
         "required": False},
        {"name": "origin_doc_annotation", "type": "boolean",
         "description": "If true: mine all queries first, then LLM-filter by origin positive; "
         "writes output_path (full mined) and output_path_origin_filtered.",
         "required": False},
        {"name": "negative_number", "type": "number",
         "description": "Target number of hard negatives per query (default 15).", "required": False},
        {"name": "range_for_sampling", "type": "string",
         "description": "Rank window for Branch A sampling, format 'start-end' (default '30-200').",
         "required": False},
        {"name": "max_queries", "type": "number",
         "description": "Cap on total queries processed; null = all.", "required": False},
    ]

    def __init__(
        self,
        task_name: str = "msmarco",
        llm_cfg: Optional[Dict] = None,
        instructions_dir: Optional[str] = None,
        concurrency_enabled: bool = False,
        max_workers: Optional[int] = None,
        jitter: float = 0.0,
        progress_logger: Optional[Any] = None,
        progress_interval_sec: float = 0.0,
    ):
        self.task_name = task_name
        self.llm_cfg = llm_cfg or {
            "model": "gpt-4.1-mini",
            "model_server": os.getenv("OPENAI_BASE_URL"),
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
        self.instructions_dir = Path(instructions_dir) if instructions_dir else INSTRUCTIONS_DIR
        self.evaluation_instruction: Optional[str] = None
        self.annotator: Optional[Assistant] = None
        self.concurrency_enabled = concurrency_enabled
        self.max_workers = max_workers
        self.jitter = jitter
        self.progress_logger = progress_logger
        self.progress_interval_sec = progress_interval_sec
        self.stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Annotation helpers (identical contract to BatchCandidateAnnotator)
    # ------------------------------------------------------------------

    def _load_instruction_for_task(self, task_name: str) -> Optional[str]:
        """Load the newest AnnoInstrRefine_* (or AnnoInstr_* fallback) for task_name."""
        instr_dir = self.instructions_dir
        if not instr_dir.exists():
            logging.error("Instructions directory not found: %s", instr_dir)
            return None
        refine_candidates = [
            p for p in instr_dir.glob(f"AnnoInstrRefine_{task_name}_*.txt")
            if "_attempt" not in p.name
        ]
        if refine_candidates:
            refine_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            path = refine_candidates[0]
        else:
            candidates = list(instr_dir.glob(f"AnnoInstr_{task_name}_*.txt"))
            if not candidates:
                logging.error(
                    "No annotation instruction for task '%s' in %s. Run AnnoInstr_Generator first.",
                    task_name, instr_dir,
                )
                return None
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            path = candidates[0]
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        logging.info("Loaded annotation instruction from %s", path.name)
        return content

    def _ensure_annotator(self, task_name: str, llm_cfg: Dict) -> None:
        """Load the annotation instruction and store llm_cfg for later per-thread use.

        We deliberately do NOT create a single shared ``self.annotator`` here so
        that concurrent threads can each create their own ``Assistant`` instance in
        ``_annotate_one`` without sharing mutable state.
        """
        if self.evaluation_instruction is not None and self.task_name == task_name and self.llm_cfg == llm_cfg:
            return
        self.task_name = task_name
        self.llm_cfg = llm_cfg
        logging.info("Loading annotation instruction for task '%s' ...", task_name)
        self.evaluation_instruction = self._load_instruction_for_task(task_name)
        if not self.evaluation_instruction:
            raise FileNotFoundError(
                f"Annotation instruction for task '{task_name}' not found under "
                f"{self.instructions_dir}. Run AnnoInstr_Generator first."
            )
        logging.info("Annotation instruction ready.")

    def _extract_json(self, text: str) -> Optional[Dict]:
        try:
            if "```json" in text:
                s = text.find("```json") + 7
                e = text.find("```", s)
                text = text[s:e].strip()
            elif "```" in text:
                s = text.find("```") + 3
                e = text.find("```", s)
                text = text[s:e].strip()
            text = text.strip().lstrip("`").rstrip("`")
            return json.loads(text)
        except Exception as e:
            logging.debug("JSON extraction failed: %s", e)
            return None

    def _annotate_one(
        self,
        query: str,
        candidates: List[Dict],
        max_retries: int = 1,
    ) -> Optional[Dict]:
        """Send one LLM annotation call for (query, candidates) and parse the result.

        A fresh ``Assistant`` is created per call so that multiple threads can
        invoke this method concurrently without sharing mutable state.
        """
        user_input = f"Query: {query}\n\nCandidate documents (each with doc_id and doc):\n\n"
        input_doc_ids: set = set()
        for c in candidates:
            doc_id = c.get("doc_id", "unknown")
            input_doc_ids.add(str(doc_id))
            doc = c.get("doc", "")
            user_input += f"[doc_id: {doc_id}]\n{doc}\n\n"
        messages = [{"role": "user", "content": user_input}]
        annotator = Assistant(llm=self.llm_cfg, system_message=self.evaluation_instruction)
        for attempt in range(max_retries + 1):
            try:
                result_text = ""
                for response in annotator.run(messages):
                    result_text = response[-1]["content"]
                annotation = self._extract_json(result_text)
                if annotation:
                    output_annotations = annotation.get("annotations", [])
                    valid_annotations = [
                        a for a in output_annotations
                        if str(a.get("doc_id")) in input_doc_ids
                    ]
                    annotation["annotations"] = valid_annotations

                    def _norm(c):
                        if c is None:
                            return ""
                        s = str(c).strip().lower()
                        if s == "positive":
                            return "positive"
                        if s in ("hard negative", "hard_negative"):
                            return "hard_negative"
                        if s in ("easy negative", "easy_negative"):
                            return "easy_negative"
                        return s

                    annotation["positive_doc_ids"] = [
                        str(a["doc_id"]) for a in valid_annotations
                        if _norm(a.get("classification")) == "positive"
                    ]
                    return annotation
                if attempt < max_retries:
                    logging.warning("Parse failed, retrying (attempt %d) ...", attempt + 1)
                    messages.append({"role": "assistant", "content": result_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your output was not valid JSON or was missing required fields. "
                            "Please output ONLY a valid JSON object with an 'annotations' list "
                            "as specified in the system instructions."
                        ),
                    })
                    self.stats["annotation_retries"] = self.stats.get("annotation_retries", 0) + 1
            except Exception as exc:
                logging.error("Error during annotation: %s", exc)
                if attempt >= max_retries:
                    break
        self.stats["annotation_parse_errors"] = self.stats.get("annotation_parse_errors", 0) + 1
        return None

    # ------------------------------------------------------------------
    # Original-doc discovery with IdAttr fallback
    # ------------------------------------------------------------------

    def _build_idattr_cache(self, idattr_path: str) -> Dict[str, Dict]:
        """Load IdAttr JSONL keyed by doc_id for O(1) fallback lookups."""
        cache: Dict[str, Dict] = {}
        try:
            with open(idattr_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    doc_id = obj.get("doc_id")
                    if doc_id is not None:
                        cache[str(doc_id)] = obj
        except Exception as exc:
            logging.warning("Could not load IdAttr file '%s': %s", idattr_path, exc)
        return cache

    def _find_original_doc(
        self,
        item: Dict,
        idattr_cache: Dict[str, Dict],
    ) -> Optional[Dict]:
        """Return a pseudo-candidate dict for the original doc (doc + doc_id only).

        score is always None here; the precise per-query cosine similarity is carried by
        q_obj["pos_score"] (injected by Query2Passage) and applied in _process_one_query_impl.

        Strategy:
          1. Find a candidate with is_original=True in any of the item's queries → correct
             doc/doc_id; score set to None (the candidate's score belongs to that other query).
          2. Fall back to the IdAttr cache using item['doc_id'].
          3. Fall back to item['doc'] / item['doc_id'] directly (always present in Q2P output).
          4. Return None only if item['doc'] is also empty (essentially never in practice).
        """
        doc_id = str(item.get("doc_id", ""))

        # Try to find is_original from any query in this item to obtain the origin doc's
        # doc and doc_id. All queries share the same origin doc so doc/doc_id are correct,
        # but the candidate's score belongs to that other query — not the current one.
        # score=None here; the actual per-query score comes from q_obj["pos_score"] injected
        # by Query2Passage and applied in _process_one_query_impl.
        for q_obj in item.get("queries", []):
            if not isinstance(q_obj, dict):
                continue
            for c in q_obj.get("candidates", []):
                if c.get("is_original"):
                    return {
                        "doc_id": c.get("doc_id", doc_id),
                        "doc": c.get("doc", ""),
                        "score": None,
                    }

        # Fallback: IdAttr cache
        if doc_id and doc_id in idattr_cache:
            idattr_obj = idattr_cache[doc_id]
            doc_content = idattr_obj.get("doc", "")
            if doc_content:
                logging.debug(
                    "Original doc %s not in top-k candidates; loaded doc from IdAttr cache.", doc_id
                )
                return {
                    "doc_id": doc_id,
                    "doc": doc_content,
                    "score": None,
                }

        # Fallback: item-level doc (always present in Q2P output JSON).
        item_doc = item.get("doc", "")
        if item_doc:
            logging.debug(
                "Original doc %s: using item-level doc as fallback (not in top-k or IdAttr).", doc_id
            )
            return {
                "doc_id": doc_id,
                "doc": item_doc,
                "score": None,
            }

        logging.warning(
            "Original doc '%s' has no doc in candidates, IdAttr cache, or item; query will be skipped.",
            doc_id,
        )
        return None

    # ------------------------------------------------------------------
    # Branch A / B negative mining
    # ------------------------------------------------------------------

    def _mine_hard_negatives(
        self,
        q_obj: Dict,
        orig_doc: Dict,
        negative_number: int,
        range_start: int,
        range_end: int,
    ) -> List[Dict]:
        """Apply Branch A or Branch B heuristic to mine hard negatives.

        Returns a list of candidate dicts (may be shorter than negative_number
        if the pool is exhausted; no corpus padding is performed).
        """
        candidates: List[Dict] = q_obj.get("candidates", [])
        # Candidates are already sorted by score desc from Q2P.

        orig_doc_content = orig_doc.get("doc", "")
        orig_doc_id = str(orig_doc.get("doc_id", ""))
        query_text = q_obj.get("query", "")

        # skip_set: doc content of known positives + the query itself (prevent trivial negs)
        skip_docs = {orig_doc_content, query_text}

        raw_score = orig_doc.get("score")
        if raw_score is None:
            # If score unknown (came from IdAttr fallback), use the highest candidate
            # score as a proxy — this biases us toward Branch B (conservative).
            raw_score = candidates[0]["score"] if candidates else 1.0
        raw_score = _score_as_float(raw_score)

        # Reference threshold: score of the min(range_end, last) ranked candidate.
        ref_idx = min(range_end - 1, len(candidates) - 1)
        ref_score = _score_as_float(candidates[ref_idx]["score"] if candidates else 0.0)

        # Exclude the original doc itself from the negative pool.
        neg_pool_candidates = [
            c for c in candidates
            if str(c.get("doc_id", "")) != orig_doc_id and c.get("doc", "") not in skip_docs
        ]

        if raw_score * 0.95 < ref_score:
            # Branch A: dense region — sample from mid-rank window [range_start, range_end).
            # The window is over the full candidate list (including original doc),
            # then we filter it from neg_pool_candidates to stay consistent.
            window = candidates[range_start:range_end]
            window_ids = {str(c.get("doc_id", "")) for c in window}
            branch_pool = [
                c for c in neg_pool_candidates
                if str(c.get("doc_id", "")) in window_ids
            ]
            if len(branch_pool) > negative_number:
                branch_pool = random.sample(branch_pool, negative_number)
            neg_selected = branch_pool
            branch = "A"
        else:
            # Branch B: sparse region — top candidates below the 0.95× threshold.
            branch_pool = [
                c for c in neg_pool_candidates
                if _score_as_float(c.get("score", 0.0)) < raw_score * 0.95
            ]
            neg_selected = branch_pool[:negative_number]
            branch = "B"

        logging.debug(
            "Query '%s...' branch=%s raw_score=%.4f ref_score=%.4f pool=%d selected=%d",
            query_text[:40], branch, raw_score, ref_score, len(branch_pool), len(neg_selected),
        )
        return neg_selected

    def _phase2_verify_origin(self, res: Dict[str, Any]) -> Dict[str, Any]:
        """Set ``origin_llm_ok`` on a Phase-1 mined result (mutates *res*)."""
        rec = res.get("record")
        if not rec:
            res["origin_llm_ok"] = False
            return res
        query_text = rec["query"]
        orig_doc_id = str(rec["pos_doc_id"])
        pos_doc = (rec.get("pos") or [""])[0]
        anno_candidate = [{
            "doc_id": orig_doc_id,
            "doc": pos_doc,
        }]
        annotation = self._annotate_one(query_text, anno_candidate)
        ok = bool(
            annotation is not None
            and orig_doc_id in annotation.get("positive_doc_ids", [])
        )
        res["origin_llm_ok"] = ok
        return res

    def _aggregate_doc_records(
        self,
        per_query_results: List[Dict[str, Any]],
        include_fn,
    ) -> tuple[Dict[str, Dict], int, int]:
        """Group accepted queries by source doc_id; *include_fn(res)* gates each row.

        Returns (output_doc_records, accepted_query_count, total_neg_count).
        """
        output_doc_records: Dict[str, Dict] = {}
        accepted_queries = 0
        total_neg_count = 0
        for res in per_query_results:
            if res.get("skip_reason"):
                continue
            if not res.get("record"):
                continue
            if not include_fn(res):
                continue
            accepted_queries += 1
            total_neg_count += int(res.get("neg_count", 0))
            item_doc_id = res["item_doc_id"]
            if item_doc_id not in output_doc_records:
                output_doc_records[item_doc_id] = {
                    "origin_doc": res["origin_doc"],
                    "doc_id": item_doc_id,
                    "queries": [],
                }
            output_doc_records[item_doc_id]["queries"].append(res["record"])
        return output_doc_records, accepted_queries, total_neg_count

    @staticmethod
    def _write_doc_jsonl(output_doc_records: Dict[str, Dict], out_path: str) -> None:
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        output_lines = [
            json.dumps(doc_rec, ensure_ascii=False)
            for doc_rec in output_doc_records.values()
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
            if output_lines:
                f.write("\n")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def call(self, params: str, **kwargs) -> str:
        params_dict = json5.loads(params)
        task_name = params_dict.get("task_name", self.task_name)
        llm_cfg = kwargs.get("llm_cfg", self.llm_cfg)
        input_path = params_dict.get("input_path") or params_dict.get("input")
        output_path = params_dict.get("output_path") or params_dict.get("output")
        _opof = params_dict.get("output_path_origin_filtered")
        if _opof is not None and str(_opof).strip():
            output_path_origin_filtered = str(_opof).strip()
        else:
            output_path_origin_filtered = None
        origin_doc_annotation = bool(params_dict.get("origin_doc_annotation", False))
        negative_number = int(params_dict.get("negative_number", 15))
        range_str = str(params_dict.get("range_for_sampling", "30-200"))
        raw_max = params_dict.get("max_queries")
        max_queries = int(raw_max) if raw_max is not None else None

        if not input_path or not str(input_path).strip():
            return json5.dumps({"error": "Missing input_path."}, ensure_ascii=False)
        if not os.path.exists(input_path):
            return json5.dumps({"error": f"Input file not found: {input_path}"}, ensure_ascii=False)

        # Parse range
        try:
            parts = range_str.split("-")
            range_start, range_end = int(parts[0]), int(parts[1])
        except Exception:
            logging.warning("Invalid range_for_sampling '%s'; using 30-200.", range_str)
            range_start, range_end = 30, 200

        if not output_path or not str(output_path).strip():
            stem = Path(input_path).stem
            output_path = f"Results/HnMine_{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        if origin_doc_annotation and not output_path_origin_filtered:
            _p = Path(output_path)
            output_path_origin_filtered = str(_p.with_name(_p.stem + "_origin_pos.jsonl"))

        # Load annotation instruction if needed
        if origin_doc_annotation:
            self._ensure_annotator(task_name, llm_cfg)

        # Build IdAttr fallback cache (path is stored at the top of the Q2P JSON)
        idattr_path = extract_input_path_prefix(input_path)
        if not idattr_path:
            try:
                with open(input_path, "rb") as _pf:
                    idattr_path = next(ijson.items(_pf, "input_path"), "") or ""
            except Exception:
                idattr_path = ""
        idattr_cache: Dict[str, Dict] = {}
        if idattr_path and os.path.exists(idattr_path):
            idattr_cache = self._build_idattr_cache(idattr_path)
            logging.info("Loaded IdAttr fallback cache: %d entries from %s",
                         len(idattr_cache), idattr_path)
        else:
            logging.warning(
                "IdAttr file '%s' not found; fallback for missing is_original candidates unavailable.",
                idattr_path,
            )

        self.stats = {
            "total_queries": 0,
            "accepted_queries": 0,
            "accepted_queries_mined": 0,
            "accepted_queries_origin_filtered": 0,
            "skipped_no_orig_doc": 0,
            "skipped_orig_not_positive": 0,
            "skipped_no_negatives": 0,
            "annotation_retries": 0,
            "annotation_parse_errors": 0,
            "branch_a_count": 0,
            "branch_b_count": 0,
            "avg_neg_count": 0.0,
            "avg_neg_count_origin_filtered": 0.0,
            "accepted_docs_all": 0,
            "accepted_docs_origin_filtered": 0,
        }

        def _process_one_query(
            item: Dict,
            q_obj: Dict,
            item_orig_doc: Optional[Dict],
        ) -> Dict[str, Any]:
            """Process a single query; always returns a result dict, never raises."""
            result: Dict[str, Any] = {
                "record": None,
                "skip_reason": None,  # "no_orig_doc" | "no_negatives" | "error"
                "branch": None,       # "a" | "b"
                "neg_count": 0,
            }
            try:
                return _process_one_query_impl(item, q_obj, item_orig_doc, result)
            except Exception as exc:
                logging.error(
                    "Unexpected error processing query '%s': %s",
                    q_obj.get("query", "")[:80], exc,
                )
                result["skip_reason"] = "error"
                return result

        def _process_one_query_impl(
            item: Dict,
            q_obj: Dict,
            item_orig_doc: Optional[Dict],
            result: Dict[str, Any],
        ) -> Dict[str, Any]:
            query_text = q_obj.get("query", "")
            candidates = q_obj.get("candidates", [])

            # 1. Resolve per-query original doc.
            # Priority: is_original candidate in this query > item_orig_doc (which itself
            # tries is_original across all queries → IdAttr cache → item["doc"]).
            # In practice query_orig_doc is None only when item["doc"] is empty.
            query_orig_doc = item_orig_doc
            for c in candidates:
                if c.get("is_original"):
                    query_orig_doc = {
                        "doc_id": c.get("doc_id", item.get("doc_id", "")),
                        "doc": c.get("doc", ""),
                        "score": c.get("score"),
                    }
                    break

            if query_orig_doc is None:
                result["skip_reason"] = "no_orig_doc"
                return result

            # Use the precise cosine score injected by Query2Passage (Step 4) when available.
            # This eliminates proxy-score bias that occurs when the origin doc falls outside top-k.
            pos_score_override = q_obj.get("pos_score")
            if pos_score_override is not None:
                query_orig_doc = {**query_orig_doc, "score": pos_score_override}

            orig_doc_content = query_orig_doc.get("doc", "")
            orig_doc_id = str(query_orig_doc.get("doc_id", ""))

            # 2. Mine hard negatives (no LLM; pure heuristic — thread-safe)
            neg_candidates = self._mine_hard_negatives(
                q_obj=q_obj,
                orig_doc=query_orig_doc,
                negative_number=negative_number,
                range_start=range_start,
                range_end=range_end,
            )

            if not neg_candidates:
                result["skip_reason"] = "no_negatives"
                return result

            # 3. Determine branch (use ``is not None`` so 0.0 from pos_score is not replaced by proxy)
            _rs = query_orig_doc.get("score")
            raw_score = _score_as_float(
                _rs if _rs is not None else (candidates[0]["score"] if candidates else 1.0)
            )
            ref_idx = min(range_end - 1, len(candidates) - 1)
            ref_score = _score_as_float(candidates[ref_idx]["score"] if candidates else 0.0)
            result["branch"] = "a" if raw_score * 0.95 < ref_score else "b"
            result["neg_count"] = len(neg_candidates)

            # 4. Build output record
            _ps_out = query_orig_doc.get("score")
            result["record"] = {
                "query": query_text,
                "pos": [orig_doc_content],
                "neg": [c.get("doc", "") for c in neg_candidates],
                "doc_id": item.get("doc_id", ""),
                "pos_doc_id": orig_doc_id,
                "neg_doc_ids": [str(c.get("doc_id", "")) for c in neg_candidates],
                "pos_score": None if _ps_out is None else _score_as_float(_ps_out),
            }
            # Metadata for doc-level grouping in the aggregation step
            result["item_doc_id"] = str(item.get("doc_id", ""))
            result["origin_doc"] = orig_doc_content
            return result

        per_query_results: List[Dict[str, Any]] = []
        remaining_cap: Optional[int] = max_queries

        try:
            if file_is_large_q2p(input_path):
                logging.info(
                    "HN mining: streaming large Q2P JSON (>=5GiB) path=%s "
                    "(concurrency=%s max_workers=%s origin_doc_annotation=%s)",
                    input_path,
                    self.concurrency_enabled,
                    self.max_workers,
                    origin_doc_annotation,
                )
                with open(input_path, "rb") as f:
                    for item in ijson.items(f, "results.item"):
                        if remaining_cap is not None and remaining_cap <= 0:
                            break
                        if not isinstance(item, dict):
                            continue
                        item_orig_doc = self._find_original_doc(item, idattr_cache)
                        local_work: List[tuple] = []
                        for q_obj in item.get("queries", []) or []:
                            if remaining_cap is not None and remaining_cap <= 0:
                                break
                            if not isinstance(q_obj, dict):
                                continue
                            query_text = q_obj.get("query", "")
                            candidates = q_obj.get("candidates", [])
                            if not query_text or not candidates:
                                continue
                            local_work.append((item, q_obj, item_orig_doc))
                            if remaining_cap is not None:
                                remaining_cap -= 1
                        if not local_work:
                            continue
                        chunk_results = run_concurrent(
                            _process_one_query,
                            [
                                {"item": a, "q_obj": b, "item_orig_doc": c}
                                for a, b, c in local_work
                            ],
                            enabled=self.concurrency_enabled,
                            max_workers=self.max_workers,
                            jitter=self.jitter,
                            progress_logger=self.progress_logger,
                            progress_label="Step 5 hard negative mining (HN)",
                            progress_interval_sec=self.progress_interval_sec,
                        )
                        per_query_results.extend(chunk_results)
            else:
                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results = data.get("results", [])
                query_work: List[tuple] = []
                for item in results:
                    if max_queries is not None and len(query_work) >= max_queries:
                        break
                    item_orig_doc = self._find_original_doc(item, idattr_cache)
                    for q_obj in item.get("queries", []) or []:
                        if max_queries is not None and len(query_work) >= max_queries:
                            break
                        if not isinstance(q_obj, dict):
                            continue
                        query_text = q_obj.get("query", "")
                        candidates = q_obj.get("candidates", [])
                        if not query_text or not candidates:
                            continue
                        query_work.append((item, q_obj, item_orig_doc))
                _n_q = len(query_work)
                logging.info(
                    "HN mining: %d queries (concurrency=%s, max_workers=%s, origin_doc_annotation=%s)",
                    _n_q,
                    self.concurrency_enabled,
                    self.max_workers,
                    origin_doc_annotation,
                )
                per_query_results = run_concurrent(
                    _process_one_query,
                    [{"item": item, "q_obj": q_obj, "item_orig_doc": orig_doc}
                     for item, q_obj, orig_doc in query_work],
                    enabled=self.concurrency_enabled,
                    max_workers=self.max_workers,
                    jitter=self.jitter,
                    progress_logger=self.progress_logger,
                    progress_label="Step 5 hard negative mining (HN)",
                    progress_interval_sec=self.progress_interval_sec,
                )
        except json.JSONDecodeError as exc:
            return json5.dumps({"error": f"Invalid Q2P JSON: {exc}"}, ensure_ascii=False)
        except Exception as exc:
            logging.error("HN mining: failed to read or parse Q2P JSON %s: %s", input_path, exc)
            return json5.dumps(
                {"error": f"Failed to read or parse Q2P JSON: {exc}"},
                ensure_ascii=False,
            )

        total_to_process = len(per_query_results)
        # ---- Phase 1 stats (heuristic mining only) ----
        with tqdm(total=total_to_process, desc="HN mining progress") as pbar:
            for res in per_query_results:
                self.stats["total_queries"] += 1
                skip = res.get("skip_reason")
                if skip == "no_orig_doc":
                    self.stats["skipped_no_orig_doc"] += 1
                elif skip == "no_negatives":
                    self.stats["skipped_no_negatives"] += 1
                elif skip == "error":
                    self.stats["skipped_errors"] = self.stats.get("skipped_errors", 0) + 1
                else:
                    branch = res.get("branch")
                    if branch == "a":
                        self.stats["branch_a_count"] += 1
                    elif branch == "b":
                        self.stats["branch_b_count"] += 1
                    self.stats["accepted_queries"] += 1
                    self.stats["accepted_queries_mined"] += 1
                pbar.update(1)

        mined_rows = [
            r for r in per_query_results
            if r.get("record") is not None and not r.get("skip_reason")
        ]

        def _p2_wrap(res: Dict[str, Any]) -> Dict[str, Any]:
            return self._phase2_verify_origin(res)

        if origin_doc_annotation and mined_rows:
            run_concurrent(
                _p2_wrap,
                [{"res": r} for r in mined_rows],
                enabled=self.concurrency_enabled,
                max_workers=self.max_workers,
                jitter=self.jitter,
                progress_logger=self.progress_logger,
                progress_label="Step 5 HN origin-doc LLM filter",
                progress_interval_sec=self.progress_interval_sec,
            )
            for r in mined_rows:
                if not r.get("origin_llm_ok"):
                    self.stats["skipped_orig_not_positive"] += 1

        # Full mined JSONL (always; equivalent to legacy origin_doc_annotation=False)
        doc_all, nq_all, tn_all = self._aggregate_doc_records(
            per_query_results, lambda _r: True
        )
        self._write_doc_jsonl(doc_all, output_path)
        self.stats["accepted_docs_all"] = len(doc_all)
        # Backward-compatible alias: full-mined doc count (legacy key).
        self.stats["accepted_docs"] = len(doc_all)
        if nq_all > 0:
            self.stats["avg_neg_count"] = round(tn_all / nq_all, 2)

        output_filtered_written: Optional[str] = None
        if origin_doc_annotation:
            doc_filt, nq_filt, tn_filt = self._aggregate_doc_records(
                per_query_results, lambda r: bool(r.get("origin_llm_ok"))
            )
            self._write_doc_jsonl(doc_filt, output_path_origin_filtered)
            output_filtered_written = output_path_origin_filtered
            self.stats["accepted_queries_origin_filtered"] = nq_filt
            self.stats["accepted_docs_origin_filtered"] = len(doc_filt)
            if nq_filt > 0:
                self.stats["avg_neg_count_origin_filtered"] = round(tn_filt / nq_filt, 2)
        else:
            self.stats["accepted_queries_origin_filtered"] = 0
            self.stats["accepted_docs_origin_filtered"] = 0

        logging.info(
            "HN mining complete. output_path=%s output_path_origin_filtered=%s stats=%s",
            output_path,
            output_filtered_written,
            self.stats,
        )
        return json5.dumps(
            {
                "output_path": output_path,
                "output_path_origin_filtered": output_filtered_written,
                "stats": self.stats,
            },
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
#  CLI entry point (mirrors Batch_Candidate_Annotator.py CLI interface)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mine hard negatives from a Query2Passage JSON file."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input Query2Passage JSON file path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL for full mined results (default: Results/HnMine_<stem>_<ts>.jsonl)")
    parser.add_argument("--output-origin-filtered", type=str, default=None,
                        dest="output_origin_filtered",
                        help="With --origin-doc-annotation: JSONL for origin-positive subset "
                             "(default: <output_stem>_origin_pos.jsonl)")
    parser.add_argument("--task", type=str, default="msmarco",
                        help="Task name (used to load annotation instruction)")
    parser.add_argument("--origin-doc-annotation", action="store_true", default=False,
                        dest="origin_doc_annotation",
                        help="Mine first, then LLM-filter by original doc; writes full + filtered JSONL")
    parser.add_argument("--negative-number", type=int, default=15,
                        dest="negative_number",
                        help="Target number of hard negatives per query (default: 15)")
    parser.add_argument("--range-for-sampling", type=str, default="30-200",
                        dest="range_for_sampling",
                        help="Branch A rank window, format 'start-end' (default: 30-200)")
    parser.add_argument("--max-queries", type=int, default=None,
                        dest="max_queries",
                        help="Max number of queries to process")
    cli_args = parser.parse_args()

    llm_cfg = {
        "model": "gpt-4.1-mini",
        "model_server": os.getenv("OPENAI_BASE_URL"),
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
    tool = HnMineTool(task_name=cli_args.task, llm_cfg=llm_cfg)
    call_params = json5.dumps({
        "input_path": cli_args.input,
        "output_path": cli_args.output,
        "output_path_origin_filtered": cli_args.output_origin_filtered,
        "task_name": cli_args.task,
        "origin_doc_annotation": cli_args.origin_doc_annotation,
        "negative_number": cli_args.negative_number,
        "range_for_sampling": cli_args.range_for_sampling,
        "max_queries": cli_args.max_queries,
    }, ensure_ascii=False)
    out = json5.loads(tool.call(call_params, llm_cfg=llm_cfg))
    if out.get("error"):
        logging.error("%s", out)
    else:
        logging.info("Wrote full mined: %s", out.get("output_path"))
        if out.get("output_path_origin_filtered"):
            logging.info("Wrote origin-filtered: %s", out["output_path_origin_filtered"])
