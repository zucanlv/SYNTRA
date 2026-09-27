"""
Synthetic Data Generation Pipeline Main Controller.

Orchestrates: corpus indexing, diverse text selection, IdAttr generation,
query instruction, diverse query generation, Query2Passage retrieval, and
batch candidate annotation. Configuration is loaded from config/config.yaml
and overridden by CLI arguments.
"""

import os
import json
import json5
import logging
import argparse
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from HighLevel_Def import HighLevel_Task_Definition
from Few_Shot_Example import Few_Shot_Example
from Corpus_Build_Index import build_index, BuildIndexConfig
from Text_Diverse_Selection import (
    select_diverse_texts,
    export_all_indexed_texts,
    DiverseSelectionConfig,
)
from QueryInstr_Generator import (
    QueryInstructionGenerator,
    evaluate_query_instruction, refine_query_instruction,
)
from AnnoInstr_Generator import (
    AnnotationInstructionGenerator,
    evaluate_annotation_instruction, refine_annotation_instruction,
)
from AnnoCalibration_Generator import (
    build_positive_only_calibration_from_fewshot,
    compare_annotations_against_calibration,
    generate_calibration_refinement_feedback,
    load_anno_calibration,
)
from IdAttr_Generator import (
    generate_idattr, IdAttrConfig, generate_idattr_for_doc,
    evaluate_idattr_outputs, refine_idattr_guidance,
)
from SelfRefine import SelfRefineLoop
from HumanReview import HumanReviewer
from qwen_agent.agents import Assistant
from DiverseQuery_Generator import DiverseQueryGenerator
from Query2Passge import Query2PassageTool
from Batch_Candidate_Annotator import BatchCandidateAnnotator
from Hn_Mine import HnMineTool
from DocType_Filter import filter_diverse_texts, FilterConfig
from Query_Filter import filter_diverse_queries, QueryFilterConfig
from concurrent_runner import run_concurrent
from llm_client_pool import configure_qwen_agent_openai_client_pool
from q2p_json_stream import (
    count_queries_q2p_streaming,
    file_is_large_q2p,
    reservoir_sample_q_objs_with_candidates,
    validate_q2p_has_nonempty_candidates,
)
from production_sample_utils import apply_production_sample
from query_length_sampler import QueryLengthSampler

# ---------------------------------------------------------------------------
#  Config loading (config.yaml + CLI merge)
# ---------------------------------------------------------------------------

def parse_optional_int(val):
    """Convert config/CLI value to int or None. None/'None'/'' -> None (greedy diverse selection); otherwise int(val)."""
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in ("none", ""):
        return None
    return int(val)


def parse_optional_str(val):
    """Convert config/CLI value to str or None. None/'None'/'' -> None."""
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in ("none", ""):
        return None
    return str(val)


def _gpu_ids_to_str(val) -> str | None:
    """Convert a list of GPU IDs from YAML to the comma-separated string used as argparse default."""
    if val is None:
        return None
    if isinstance(val, list):
        return ",".join(str(x) for x in val)
    return str(val)  # already a string


def load_config(config_path: str = None) -> dict:
    """Load default config from config/config.yaml. Returns dict; missing file returns {}."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
    path = Path(config_path)
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        logging.warning("Failed to load config from %s: %s", path, e)
        return {}


def resolve_config_yaml_path(args_config: str | None) -> Path:
    """Path to the YAML file used for this run (default: config/config.yaml)."""
    if args_config:
        return Path(args_config).expanduser().resolve()
    return Path(__file__).resolve().parent / "config" / "config.yaml"


def materialize_step4_diverse_query_input(src_path: str, results_dir_run: str, logger) -> str | None:
    """Copy a DiverseQuery batch JSON into the run directory for Query2Passage.

    Expects the same schema as Step 3 output: a JSON object with a non-empty ``results`` list.
    Returns the path to the copy, or None on error.
    """
    p = Path(src_path).expanduser()
    if not p.is_file():
        logger.error("[Step4] diverse query batch file not found: %s", p)
        return None
    if p.suffix.lower() != ".json":
        logger.error("[Step4] expected a .json file (DiverseQuery batch with 'results'): %s", p)
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("[Step4] invalid JSON in %s: %s", p, exc)
        return None
    results = data.get("results")
    if not isinstance(results, list) or len(results) == 0:
        logger.error("[Step4] JSON must contain a non-empty 'results' array: %s", p)
        return None
    base = Path(__file__).resolve().parent
    out_dir = base / results_dir_run if not Path(results_dir_run).is_absolute() else Path(results_dir_run)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = out_dir / f"DiverseQuery_FromFile_{ts}.json"
    shutil.copy2(p, dest)
    logger.info(
        "[Step4] DiverseQuery batch: source=%s copy=%s (results items=%d)",
        p,
        dest,
        len(results),
    )
    return str(dest)


def materialize_step5_query2passage_input(src_path: str, results_dir_run: str, logger) -> str | None:
    """Prepare Query2Passage JSON for Step 5 (path returned to the pipeline).

    Expects the same schema as Query2Passage tool output: non-empty ``results``,
    and at least one query **dict** with a non-empty ``candidates`` list.

    Small files: validated via ``json.load``, then copied into ``results_dir_run``.
    Large files (see ``q2p_json_stream.LARGE_Q2P_BYTES``): streaming validation only;
    returns the **resolved source path** (no copy) to avoid time and disk duplication.
    """
    p = Path(src_path).expanduser()
    if not p.is_file():
        logger.error("[Step5] Query2Passage input file not found: %s", p)
        return None
    if p.suffix.lower() != ".json":
        logger.error("[Step5] expected a .json file (Query2Passage output): %s", p)
        return None

    if file_is_large_q2p(p):
        try:
            ok_candidates = validate_q2p_has_nonempty_candidates(p)
        except Exception as exc:
            logger.error("[Step5] streaming validation failed (invalid JSON or read error): %s: %s", p, exc)
            return None
        if not ok_candidates:
            logger.error(
                "[Step5] no query with non-empty 'candidates' found (streaming scan; "
                "need Query2Passage output, not a raw DiverseQuery batch): %s",
                p,
            )
            return None
        dest_str = str(p.resolve())
        logger.info(
            "[Step5] Large Query2Passage JSON (>=5GiB): skip full parse and file copy; "
            "using source path=%s",
            dest_str,
        )
        return dest_str

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("[Step5] invalid JSON in %s: %s", p, exc)
        return None
    results = data.get("results")
    if not isinstance(results, list) or len(results) == 0:
        logger.error("[Step5] JSON must contain a non-empty 'results' array: %s", p)
        return None
    has_candidates = False
    for item in results:
        if not isinstance(item, dict):
            continue
        for q_obj in item.get("queries", []):
            if isinstance(q_obj, dict):
                cands = q_obj.get("candidates")
                if isinstance(cands, list) and len(cands) > 0:
                    has_candidates = True
                    break
        if has_candidates:
            break
    if not has_candidates:
        logger.error(
            "[Step5] no query with non-empty 'candidates' found (need Query2Passage output, "
            "not a raw DiverseQuery batch): %s",
            p,
        )
        return None
    base = Path(__file__).resolve().parent
    out_dir = base / results_dir_run if not Path(results_dir_run).is_absolute() else Path(results_dir_run)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = out_dir / f"Query2Passage_FromFile_{ts}.json"
    shutil.copy2(p, dest)
    logger.info(
        "[Step5] Query2Passage input: source=%s copy=%s (results items=%d)",
        p,
        dest,
        len(results),
    )
    return str(dest)


def count_queries_in_q2p_json(data: dict) -> int:
    """Count queries in Query2Passage-style JSON (same flattening idea as Query2Passge)."""
    n = 0
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        for q_item in item.get("queries", []):
            if isinstance(q_item, str):
                if q_item.strip():
                    n += 1
            elif isinstance(q_item, dict) and q_item.get("query"):
                n += 1
    return n


def get_doc_text_for_generation(item: dict) -> str:
    """Return the document text field used by IdAttr/DiverseQuery inputs."""
    doc = item.get("doc")
    if isinstance(doc, str) and doc.strip():
        return doc.strip()
    title = item.get("title")
    text = item.get("text")
    parts = [x.strip() for x in (title, text) if isinstance(x, str) and x.strip()]
    return "\n".join(parts)


def get_llm_cfg_for_stage(stage: str, global_llm_cfg: dict, config: dict) -> dict:
    """Merge global LLM config with per-stage overrides from config['llm_stages'].

    stage: one of idattr_refine | idattr_gen | queryinstr | diversequery | annoinstr | annotation | filter | queryfilter
    global_llm_cfg: dict with keys model, model_server, api_key (from CLI/env).
    config: full config dict from load_config() (may contain llm_stages).
    Returns a new dict; global_llm_cfg is not mutated.
    """
    out = dict(global_llm_cfg)
    stages = config.get("llm_stages") or {}
    stage_cfg = stages.get(stage) if isinstance(stages, dict) else None
    if isinstance(stage_cfg, dict):
        if stage_cfg.get("model") is not None:
            out["model"] = stage_cfg["model"]
        if stage_cfg.get("model_server") is not None:
            out["model_server"] = stage_cfg["model_server"]
        if stage_cfg.get("api_key") is not None:
            out["api_key"] = stage_cfg["api_key"]
    return out


# ---------------------------------------------------------------------------
#  Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str):
    """Configure logging to write to log_dir (e.g. Logs/{task_name}_{time_stamp}/)."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


@contextmanager
def periodic_progress_log(logger, label: str, interval_sec: float):
    """Emit ``[Progress] <label> — still running ...`` on *logger* every *interval_sec* until the block exits.

    If *interval_sec* <= 0, the context manager is a no-op (no background thread).
    """
    if interval_sec <= 0:
        yield
        return
    stop = threading.Event()

    def _heartbeat():
        while not stop.is_set():
            if stop.wait(timeout=interval_sec):
                break
            logger.info("[Progress] %s — still running ...", label)

    t = threading.Thread(target=_heartbeat, daemon=True, name=f"hb-{label[:32]}")
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=2.0)


@contextmanager
def optional_file_lock(lock_path: str | None, logger, label: str):
    """Optionally serialize access to an external resource across processes."""
    if not lock_path:
        yield
        return

    import fcntl

    path = Path(lock_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as lock_f:
        logger.info("Waiting for %s lock: %s", label, path)
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        logger.info("Acquired %s lock: %s", label, path)
        try:
            yield
        finally:
            logger.info("Releasing %s lock: %s", label, path)


# ---------------------------------------------------------------------------
#  IdAttr self-refine helpers
# ---------------------------------------------------------------------------

_REFINE_CACHE_DIR = Path(__file__).resolve().parent / "Results" / "Instructions"
_INSTRUCTIONS_STORE_DIR = Path(__file__).resolve().parent / "Results" / "Instructions"


def _load_cached_idattr_refinement(task: str, logger) -> str | None:
    """Load the latest cached refinement guidance for *task* from the store dir.

    Only considers files whose names contain 'Refine' and exclude per-attempt
    and history files (_attempt, _history) so we load the final guidance only.

    Returns:
    - str: cached guidance content (can be empty, meaning no extra refinement)
    - None: no cached final guidance file found
    """
    _INSTRUCTIONS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        (p for p in _INSTRUCTIONS_STORE_DIR.glob(f"IdAttrRefine_{task}_*.txt")
         if "_attempt" not in p.name and "_history" not in p.name),
        reverse=True,
    )
    if candidates:
        path = candidates[0]
        with open(path, "r", encoding="utf-8") as f:
            guidance = f.read().strip()
        logger.info("Loaded cached IdAttr refinement guidance from %s (len=%d)",
                     path, len(guidance))
        return guidance
    logger.info("No cached IdAttr refinement guidance found for task '%s'; "
                "cannot continue in production mode.", task)
    return None


def _store_idattr_refinement(task: str, guidance: str, logger) -> Path:
    """Store final IdAttr refinement guidance in the cross-run instruction store."""
    _INSTRUCTIONS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _INSTRUCTIONS_STORE_DIR / f"IdAttrRefine_{task}_{ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(guidance)
    logger.info("Stored IdAttr refinement guidance to store dir %s", path)
    return path


def _cache_idattr_refinement_history(task: str, result, logger) -> list[Path]:
    """Persist each attempt's refinement guidance and a combined history file.

    Writes:
    - IdAttrRefine_{task}_{ts}_attempt_01.txt, _attempt_02.txt, ... (per-attempt guidance)
    - IdAttrRefine_{task}_{ts}_history.txt (combined: attempt, passed, feedback, guidance)
    - IdAttrRefine_{task}_{ts}.txt (final guidance)
    """
    from SelfRefine import RefineResult

    _REFINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []

    if not isinstance(result, RefineResult) or not result.history:
        return paths

    for rec in result.history:
        n = rec.attempt
        path = _REFINE_CACHE_DIR / f"IdAttrRefine_{task}_{ts}_attempt_{n:02d}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(rec.artifact if rec.artifact else "")
        paths.append(path)
        logger.info("Cached IdAttr refinement attempt %d to %s", n, path)

    history_path = _REFINE_CACHE_DIR / f"IdAttrRefine_{task}_{ts}_history.txt"
    with open(history_path, "w", encoding="utf-8") as f:
        for rec in result.history:
            f.write(f"{'='*60}\nAttempt {rec.attempt}  passed={rec.passed}\n")
            f.write("Guidance:\n")
            f.write((rec.artifact if rec.artifact else "(empty)") + "\n\n")
            f.write(f"Feedback: {rec.feedback}\n\n")
        f.write(f"{'='*60}\nFinal: passed={result.passed} attempt={result.attempt}\n")
    paths.append(history_path)
    logger.info("Cached IdAttr refinement history to %s", history_path)

    final_path = _REFINE_CACHE_DIR / f"IdAttrRefine_{task}_{ts}.txt"
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(result.artifact if result.artifact else "")
    paths.append(final_path)
    logger.info("Cached IdAttr refinement final guidance to %s", final_path)
    return paths


def _run_idattr_self_refine(
    diverse_texts_path: str,
    task: str,
    llm_cfg: dict,
    limit: int | None,
    seed: int | None,
    logger,
    human_reviewer: HumanReviewer | None = None,
    llm_cfg_gen: dict | None = None,
) -> str:
    """Run the self-refine loop on sample documents to calibrate the IdAttr prompt.

    llm_cfg_gen: LLM config used for IdAttr *generation* inside the loop
                 . Falls back to llm_cfg when None.
    llm_cfg:     LLM config used for *evaluation* and *refinement* steps
                 (e.g. a stronger external model).

    Returns the final refinement guidance string (may be empty if the
    original prompt already passes evaluation).
    """
    import random as _random

    _llm_cfg_gen = llm_cfg_gen if llm_cfg_gen is not None else llm_cfg

    # Load sample documents
    docs = []
    with open(diverse_texts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except Exception:
                    pass

    if limit is not None and limit < len(docs):
        if seed is not None:
            _random.seed(seed)
        docs = _random.sample(docs, limit)
    logger.info("[SelfRefine] Sample docs for calibration: %d", len(docs))

    def generate_fn(guidance: str):
        """Run IdAttr generation on sample docs with the given guidance."""
        results = []
        for i, doc in enumerate(docs):
            logger.info("[SelfRefine] Generating IdAttr for sample doc %d/%d ...",
                        i + 1, len(docs))
            result = generate_idattr_for_doc(
                doc=doc,
                llm_cfg=_llm_cfg_gen,
                task_name=task,
                refinement_guidance=guidance,
                logger=logger,
            )
            results.append(result)
        return results

    def evaluate_fn(outputs):
        """LLM-based evaluation of IdAttr outputs."""
        return evaluate_idattr_outputs(
            results=outputs,
            task_name=task,
            llm_cfg=llm_cfg,
            logger=logger,
        )

    prompt_hook = human_reviewer.review_refine_prompt if human_reviewer else None

    def refine_fn(guidance, outputs, feedback):
        """LLM-based prompt refinement (with optional human prompt review)."""
        return refine_idattr_guidance(
            current_guidance=guidance,
            results=outputs,
            feedback=feedback,
            task_name=task,
            llm_cfg=llm_cfg,
            logger=logger,
            prompt_hook=prompt_hook,
        )

    loop = SelfRefineLoop(
        max_attempts=3,
        logger=logger,
        human_reviewer=human_reviewer,
        label="IdAttr",
    )
    result = loop.run(
        initial_artifact="",
        generate_fn=generate_fn,
        evaluate_fn=evaluate_fn,
        refine_fn=refine_fn,
    )

    guidance = result.artifact or ""
    _cache_idattr_refinement_history(task, result, logger)
    _store_idattr_refinement(task, guidance, logger)
    logger.info(
        "[SelfRefine] Finished: passed=%s attempt=%d guidance_len=%d",
        result.passed, result.attempt, len(guidance),
    )
    return guidance


# ---------------------------------------------------------------------------
#  Query instruction self-refine helpers
# ---------------------------------------------------------------------------


def _load_cached_query_instruction(task: str, logger) -> str | None:
    """Load the latest refined query instruction from the store dir, or return None."""
    _INSTRUCTIONS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    # Exclude per-attempt and history files so we load only the final instruction
    candidates = sorted(
        (p for p in _INSTRUCTIONS_STORE_DIR.glob(f"QueryInstrRefine_{task}_*.txt")
         if "_attempt" not in p.name and "_history" not in p.name),
        reverse=True,
    )
    if candidates:
        path = candidates[0]
        with open(path, "r", encoding="utf-8") as f:
            instr = f.read().strip()
        logger.info("Loaded cached refined query instruction from %s (len=%d)",
                     path, len(instr))
        return instr
    logger.info("No cached refined query instruction for task '%s'.", task)
    return None


def _cache_length_sampler(task: str, sampler: "QueryLengthSampler", logger) -> Path:
    """Save QueryLengthDist to the per-run cache dir (_REFINE_CACHE_DIR)."""
    _REFINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _REFINE_CACHE_DIR / f"QueryLengthDist_{task}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sampler.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info("Cached QueryLengthDist to per-run dir %s", path)
    return path


def _store_length_sampler(task: str, sampler: "QueryLengthSampler", logger) -> Path:
    """Store QueryLengthDist in the cross-run instruction store (_INSTRUCTIONS_STORE_DIR)."""
    _INSTRUCTIONS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _INSTRUCTIONS_STORE_DIR / f"QueryLengthDist_{task}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sampler.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info("Stored QueryLengthDist to store dir %s", path)
    return path


def _load_cached_length_sampler(task: str, logger) -> "QueryLengthSampler | None":
    """Load the latest QueryLengthDist_{task}_*.json from the cross-run store dir.

    Returns a ready-to-use ``QueryLengthSampler``, or ``None`` if no cache exists.
    Mirrors the pattern of ``_load_cached_query_instruction``.
    """
    _INSTRUCTIONS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        _INSTRUCTIONS_STORE_DIR.glob(f"QueryLengthDist_{task}_*.json"),
        reverse=True,
    )
    if candidates:
        path = candidates[0]
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sampler = QueryLengthSampler.from_dict(data, task)
        logger.info("Loaded cached QueryLengthDist from %s  (%s)", path, sampler)
        return sampler
    logger.info("No cached QueryLengthDist for task '%s'.", task)
    return None


def _cache_query_instruction(task: str, instruction: str, logger) -> Path:
    """Persist final query instruction to the per-run cache dir."""
    _REFINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _REFINE_CACHE_DIR / f"QueryInstrRefine_{task}_{ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(instruction)
    logger.info("Cached refined query instruction to %s", path)
    return path


def _store_query_instruction(task: str, instruction: str, logger) -> Path:
    """Store final query instruction in the cross-run instruction store."""
    _INSTRUCTIONS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _INSTRUCTIONS_STORE_DIR / f"QueryInstrRefine_{task}_{ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(instruction)
    logger.info("Stored query instruction to store dir %s", path)
    return path


def _cache_query_instruction_attempt(
    task: str, attempt: int, instruction: str, run_ts: str, logger
) -> Path:
    """Save one attempt's query instruction for history (filename includes attempt number)."""
    _REFINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _REFINE_CACHE_DIR / f"QueryInstrRefine_{task}_attempt{attempt}_{run_ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(instruction)
    logger.info("Cached query instruction attempt %d to %s", attempt, path)
    return path


def _cache_query_instruction_refinement_history(
    task: str, result, run_ts: str, logger
) -> Path | None:
    """Persist query instruction refinement history: each attempt's instruction and feedback.

    Writes QueryInstrRefine_{task}_{run_ts}_history.txt (same format as IdAttr history).
    """
    from SelfRefine import RefineResult

    _REFINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not isinstance(result, RefineResult) or not result.history:
        return None
    history_path = _REFINE_CACHE_DIR / f"QueryInstrRefine_{task}_{run_ts}_history.txt"
    with open(history_path, "w", encoding="utf-8") as f:
        for rec in result.history:
            f.write(f"{'='*60}\nAttempt {rec.attempt}  passed={rec.passed}\n")
            f.write("Instruction:\n")
            f.write((rec.artifact if rec.artifact else "(empty)") + "\n\n")
            f.write(f"Feedback: {rec.feedback}\n\n")
        f.write(f"{'='*60}\nFinal: passed={result.passed} attempt={result.attempt}\n")
    logger.info("Cached query instruction refinement history to %s", history_path)
    return history_path


def _run_query_instr_self_refine(
    idattr_results_path: str,
    task: str,
    llm_cfg_queryinstr: dict,
    llm_cfg_diversequery: dict,
    queries_per_passage: int,
    logger,
    human_reviewer: HumanReviewer | None = None,
    use_idattr: bool = True,
    limit: int | None = None,
    seed: int | None = None,
) -> str:
    """Self-refine the query instruction using sample query generation.

    Returns the final (possibly refined) query instruction string.
    """
    with open(idattr_results_path, "r", encoding="utf-8") as f:
        id_attr_lines = [line.strip() for line in f if line.strip()]
    sample_docs = []
    for line in id_attr_lines:
        item = json.loads(line)
        if use_idattr:
            sample_docs.append(item)
        else:
            sample_docs.append({
                "doc_id": item.get("doc_id", ""),
                "doc": get_doc_text_for_generation(item),
                "identifiers": [],
                "attributes": [],
            })
    if limit is not None and limit < len(sample_docs):
        import random as _random
        if seed is not None:
            _random.seed(seed)
        sample_docs = _random.sample(sample_docs, limit)
        logger.info(
            "[SelfRefine-QueryInstr] Sampled %d doc-only items (seed=%s).",
            len(sample_docs), seed,
        )

    initial_res = json5.loads(
        QueryInstructionGenerator().call(
            json5.dumps({
                "task_name": task,
                "force_generate": True,
            }),
            llm_cfg=llm_cfg_queryinstr,
        )
    )
    initial_instruction = initial_res["generated_instruction"]

    dq_tool = DiverseQueryGenerator()

    def generate_fn(instruction: str):
        # Returns List[{"doc": str, "query": str}] so that evaluate_fn and
        # refine_fn can see the source document alongside the generated query.
        query_doc_pairs = []
        for i, it in enumerate(sample_docs):
            logger.info("[SelfRefine-QueryInstr] Generating queries for doc %d/%d ...",
                        i + 1, len(sample_docs))
            res_text = dq_tool.call(
                json5.dumps({
                    "doc": it["doc"],
                    "identifiers": it["identifiers"],
                    "attributes": it["attributes"],
                    "num_queries": queries_per_passage,
                    "task_instruction": instruction,
                    "task_name": task,
                    "use_idattr": use_idattr,
                }),
                llm_cfg=llm_cfg_diversequery,
            )
            dq_result = json5.loads(res_text)
            for q in dq_result.get("queries", []):
                if isinstance(q, dict) and isinstance(q.get("query"), str):
                    query_doc_pairs.append({"doc": it["doc"], "query": q["query"]})
        return query_doc_pairs

    _state = {"current_instruction": initial_instruction}

    def _generate_fn_wrapper(instruction: str):
        _state["current_instruction"] = instruction
        return generate_fn(instruction)

    def evaluate_fn(query_doc_pairs):
        return evaluate_query_instruction(
            query_instruction=_state["current_instruction"],
            query_doc_pairs=query_doc_pairs,
            task_name=task,
            llm_cfg=llm_cfg_queryinstr,
            logger=logger,
        )

    prompt_hook = human_reviewer.review_refine_prompt if human_reviewer else None

    def refine_fn(instruction, query_doc_pairs, feedback):
        return refine_query_instruction(
            query_instruction=instruction,
            query_doc_pairs=query_doc_pairs,
            feedback=feedback,
            task_name=task,
            llm_cfg=llm_cfg_queryinstr,
            logger=logger,
            prompt_hook=prompt_hook,
        )

    loop = SelfRefineLoop(
        max_attempts=3,
        logger=logger,
        human_reviewer=human_reviewer,
        label="QueryInstr",
    )
    result = loop.run(
        initial_artifact=initial_instruction,
        generate_fn=_generate_fn_wrapper,
        evaluate_fn=evaluate_fn,
        refine_fn=refine_fn,
    )

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for entry in result.history:
        _cache_query_instruction_attempt(
            task, entry.attempt, entry.artifact, run_ts, logger
        )
    _cache_query_instruction_refinement_history(task, result, run_ts, logger)

    instruction = result.artifact or initial_instruction
    _cache_query_instruction(task, instruction, logger)
    _store_query_instruction(task, instruction, logger)
    logger.info(
        "[SelfRefine-QueryInstr] Finished: passed=%s attempt=%d len=%d",
        result.passed, result.attempt, len(instruction),
    )
    return instruction


# ---------------------------------------------------------------------------
#  Annotation instruction self-refine helpers
# ---------------------------------------------------------------------------


def _load_cached_anno_instruction(task: str, logger) -> str | None:
    """Load the latest refined annotation instruction from the store dir, or return None.

    Excludes per-attempt and history files so we load only the final refined
    instruction (filename contains Refine, no attempt, no _history).
    """
    _INSTRUCTIONS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        (p for p in _INSTRUCTIONS_STORE_DIR.glob(f"AnnoInstrRefine_{task}_*.txt")
         if "_attempt" not in p.name and "_history" not in p.name),
        reverse=True,
    )
    if candidates:
        path = candidates[0]
        with open(path, "r", encoding="utf-8") as f:
            instr = f.read().strip()
        logger.info("Loaded cached refined annotation instruction from %s (len=%d)",
                     path, len(instr))
        return instr
    logger.info("No cached refined annotation instruction for task '%s'.", task)
    return None


def _cache_anno_instruction(task: str, instruction: str, logger) -> Path:
    """Persist final annotation instruction to the per-run cache dir."""
    _REFINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _REFINE_CACHE_DIR / f"AnnoInstrRefine_{task}_{ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(instruction)
    logger.info("Cached refined annotation instruction to %s", path)
    return path


def _store_anno_instruction(task: str, instruction: str, logger) -> Path:
    """Store final annotation instruction in the cross-run instruction store."""
    _INSTRUCTIONS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _INSTRUCTIONS_STORE_DIR / f"AnnoInstrRefine_{task}_{ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(instruction)
    logger.info("Stored annotation instruction to store dir %s", path)
    return path


def _cache_anno_instruction_attempt(
    task: str, attempt: int, instruction: str, run_ts: str, logger
) -> Path:
    """Save one attempt's annotation instruction for history (filename includes attempt number)."""
    _REFINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _REFINE_CACHE_DIR / f"AnnoInstrRefine_{task}_attempt{attempt}_{run_ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(instruction)
    logger.info("Cached annotation instruction attempt %d to %s", attempt, path)
    return path


def _cache_anno_instruction_refinement_history(
    task: str, result, run_ts: str, logger
) -> Path | None:
    """Persist annotation instruction refinement history: each attempt's instruction and feedback.

    Writes AnnoInstrRefine_{task}_{run_ts}_history.txt (same format as IdAttr history).
    """
    from SelfRefine import RefineResult

    _REFINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not isinstance(result, RefineResult) or not result.history:
        return None
    history_path = _REFINE_CACHE_DIR / f"AnnoInstrRefine_{task}_{run_ts}_history.txt"
    with open(history_path, "w", encoding="utf-8") as f:
        for rec in result.history:
            f.write(f"{'='*60}\nAttempt {rec.attempt}  passed={rec.passed}\n")
            f.write("Instruction:\n")
            f.write((rec.artifact if rec.artifact else "(empty)") + "\n\n")
            f.write(f"Feedback: {rec.feedback}\n\n")
        f.write(f"{'='*60}\nFinal: passed={result.passed} attempt={result.attempt}\n")
    logger.info("Cached annotation instruction refinement history to %s", history_path)
    return history_path


def _write_anno_instruction_to_cache(task: str, instruction: str):
    """Write annotation instruction as AnnoInstr_{task}_*.txt (legacy format).

    BatchCandidateAnnotator prefers AnnoInstrRefine_* first; use this only if
    you need to write the non-Refine naming explicitly."""
    import hashlib
    query_type, doc_type, relevance_def = HighLevel_Task_Definition[task]
    def_content = f"{query_type}|{doc_type}|{relevance_def}"
    def_hash = hashlib.md5(def_content.encode("utf-8")).hexdigest()[:8]
    _REFINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _REFINE_CACHE_DIR / f"AnnoInstr_{task}_{def_hash}_{ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(instruction)
    return path


def _run_anno_instr_self_refine(
    q2p_output_path: str,
    task: str,
    llm_cfg_annoinstr: dict,
    llm_cfg_annotation: dict,
    logger,
    sample_size: int = 3,
    human_reviewer: HumanReviewer | None = None,
    candidates_num_each_anno: int | None = None,
    hide_doc_id_for_single_doc: bool = False,
    annotation_prompt_mode: str = "multi_doc_with_doc_id",
    max_attempts: int = 3,
) -> str:
    """Self-refine the annotation instruction using sample annotations.

    Returns the final (possibly refined) annotation instruction string.
    """
    import random as _random

    if file_is_large_q2p(q2p_output_path):
        sample_query_objs = reservoir_sample_q_objs_with_candidates(
            q2p_output_path, sample_size, _random
        )
        # Same as the small-file branch below: empty pool → continue (no sample queries).
    else:
        with open(q2p_output_path, "r", encoding="utf-8") as f:
            q2p_data = json.load(f)

        all_query_objs = []
        for item in q2p_data.get("results", []):
            for q_obj in item.get("queries", []):
                if isinstance(q_obj, dict) and q_obj.get("query") and q_obj.get("candidates"):
                    all_query_objs.append(q_obj)

        if len(all_query_objs) > sample_size:
            sample_query_objs = _random.sample(all_query_objs, sample_size)
        else:
            sample_query_objs = all_query_objs
    logger.info("[SelfRefine-AnnoInstr] Sample queries for calibration: %d", len(sample_query_objs))

    initial_res = json5.loads(
        AnnotationInstructionGenerator().call(
            json5.dumps({
                "task_name": task,
                "force_generate": True,
                "annotation_prompt_mode": annotation_prompt_mode,
            }),
            llm_cfg=llm_cfg_annoinstr,
        )
    )
    initial_instruction = initial_res["generated_instruction"]

    def generate_fn(instruction: str):
        annotator = BatchCandidateAnnotator(task_name=task, llm_cfg=llm_cfg_annotation)
        annotator.evaluation_instruction = instruction
        annotator.annotator = Assistant(llm=llm_cfg_annotation, system_message=instruction)

        annotation_samples = []
        for i, q_obj in enumerate(sample_query_objs):
            logger.info("[SelfRefine-AnnoInstr] Annotating sample query %d/%d ...",
                        i + 1, len(sample_query_objs))
            result = annotator._annotate_one(
                q_obj["query"],
                q_obj["candidates"],
                candidates_num_each_anno=candidates_num_each_anno,
                hide_doc_id_for_single_doc=hide_doc_id_for_single_doc,
            )
            sample = {
                "query": q_obj["query"],
                "candidates": [
                    {"doc_id": c.get("doc_id", ""), "doc": (c.get("doc") or "")[:300]}
                    for c in q_obj["candidates"][:5] # why top 5????????
                ],
            }
            if result:
                sample["annotations"] = result.get("annotations", [])
                sample["positive_doc_ids"] = result.get("positive_doc_ids", [])
                sample["hard_negative_doc_ids"] = result.get("hard_negative_doc_ids", [])
                sample["easy_negative_doc_ids"] = result.get("easy_negative_doc_ids", [])
            else:
                sample["annotations"] = []
                sample["positive_doc_ids"] = []
                sample["hard_negative_doc_ids"] = []
                sample["easy_negative_doc_ids"] = []
            annotation_samples.append(sample)
        return annotation_samples

    _state = {"current_instruction": initial_instruction}

    def _generate_fn_wrapper(instruction: str):
        _state["current_instruction"] = instruction
        return generate_fn(instruction)

    def evaluate_fn(annotation_samples):
        return evaluate_annotation_instruction(
            annotation_instruction=_state["current_instruction"],
            annotation_samples=annotation_samples,
            task_name=task,
            llm_cfg=llm_cfg_annoinstr,
            logger=logger,
            annotation_prompt_mode=annotation_prompt_mode,
        )

    prompt_hook = human_reviewer.review_refine_prompt if human_reviewer else None

    def refine_fn(instruction, annotation_samples, feedback):
        return refine_annotation_instruction(
            annotation_instruction=instruction,
            annotation_samples=annotation_samples,
            feedback=feedback,
            task_name=task,
            llm_cfg=llm_cfg_annoinstr,
            logger=logger,
            prompt_hook=prompt_hook,
            annotation_prompt_mode=annotation_prompt_mode,
        )

    loop = SelfRefineLoop(
        max_attempts=max_attempts,
        logger=logger,
        human_reviewer=human_reviewer,
        label="AnnoInstr",
    )
    result = loop.run(
        initial_artifact=initial_instruction,
        generate_fn=_generate_fn_wrapper,
        evaluate_fn=evaluate_fn,
        refine_fn=refine_fn,
    )

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for entry in result.history:
        _cache_anno_instruction_attempt(
            task, entry.attempt, entry.artifact, run_ts, logger
        )
    _cache_anno_instruction_refinement_history(task, result, run_ts, logger)

    instruction = result.artifact or initial_instruction
    _cache_anno_instruction(task, instruction, logger)
    _store_anno_instruction(task, instruction, logger)
    logger.info(
        "[SelfRefine-AnnoInstr] Finished: passed=%s attempt=%d len=%d",
        result.passed, result.attempt, len(instruction),
    )
    return instruction


def _run_anno_instr_self_refine_from_calibration(
    anno_calibration_path: str | None,
    task: str,
    llm_cfg_annoinstr: dict,
    llm_cfg_annotation: dict,
    logger,
    human_reviewer: HumanReviewer | None = None,
    candidates_num_each_anno: int | None = None,
    hide_doc_id_for_single_doc: bool = False,
    annotation_prompt_mode: str = "single_doc_no_doc_id",
    max_attempts: int = 3,
    calibration_data: dict | None = None,
    calibration_source_label: str = "Calib",
    calibration_label_granularity: str = "three_way",
) -> str:
    """Self-refine annotation instruction using a confirmed calibration set.

    The calibration set is a task-level artifact with golden queries and
    human-confirmed positive/hard/easy candidates. Evaluation is deterministic:
    predicted labels are compared against ``expected_label`` instead of asking
    an LLM to judge correctness from scratch.
    """
    if calibration_data is not None:
        calibration = calibration_data
        calibration_ref = calibration.get("source") or calibration_source_label
    else:
        if not anno_calibration_path:
            raise ValueError("anno_calibration_path is required when calibration_data is not provided")
        calibration = load_anno_calibration(anno_calibration_path)
        calibration_ref = anno_calibration_path

    calibration_samples = calibration["samples"]
    logger.info(
        "[SelfRefine-AnnoInstr-%s] Loaded calibration set: %s samples=%d",
        calibration_source_label, calibration_ref, len(calibration_samples),
    )

    initial_res = json5.loads(
        AnnotationInstructionGenerator().call(
            json5.dumps({
                "task_name": task,
                "force_generate": True,
                "annotation_prompt_mode": annotation_prompt_mode,
            }),
            llm_cfg=llm_cfg_annoinstr,
        )
    )
    initial_instruction = initial_res["generated_instruction"]

    def generate_fn(instruction: str):
        annotator = BatchCandidateAnnotator(task_name=task, llm_cfg=llm_cfg_annotation)
        annotator.evaluation_instruction = instruction

        annotation_samples = []
        for i, sample in enumerate(calibration_samples):
            logger.info(
                "[SelfRefine-AnnoInstr-%s] Annotating calibration query %d/%d ...",
                calibration_source_label, i + 1, len(calibration_samples),
            )
            candidates = sample.get("candidates", [])
            result = annotator._annotate_one(
                sample["query"],
                candidates,
                candidates_num_each_anno=candidates_num_each_anno,
                hide_doc_id_for_single_doc=hide_doc_id_for_single_doc,
            )
            out = {
                "query_id": sample.get("query_id") or sample.get("query"),
                "query": sample["query"],
                "candidates": candidates,
            }
            if result:
                out["annotations"] = result.get("annotations", [])
                out["positive_doc_ids"] = result.get("positive_doc_ids", [])
                out["hard_negative_doc_ids"] = result.get("hard_negative_doc_ids", [])
                out["easy_negative_doc_ids"] = result.get("easy_negative_doc_ids", [])
            else:
                out["annotations"] = []
                out["positive_doc_ids"] = []
                out["hard_negative_doc_ids"] = []
                out["easy_negative_doc_ids"] = []
            annotation_samples.append(out)
        return annotation_samples

    _state = {"current_instruction": initial_instruction}

    def _generate_fn_wrapper(instruction: str):
        _state["current_instruction"] = instruction
        return generate_fn(instruction)

    def evaluate_fn(annotation_samples):
        passed, feedback, details = compare_annotations_against_calibration(
            annotation_samples=annotation_samples,
            calibration_samples=calibration_samples,
            label_granularity=calibration_label_granularity,
        )
        if not passed:
            feedback = generate_calibration_refinement_feedback(
                annotation_instruction=_state["current_instruction"],
                task_name=task,
                llm_cfg=llm_cfg_annoinstr,
                compare_feedback=feedback,
                compare_details=details,
                logger=logger,
            )
        return passed, feedback, details

    prompt_hook = human_reviewer.review_refine_prompt if human_reviewer else None

    def refine_fn(instruction, annotation_samples, feedback):
        return refine_annotation_instruction(
            annotation_instruction=instruction,
            annotation_samples=annotation_samples,
            feedback=feedback,
            task_name=task,
            llm_cfg=llm_cfg_annoinstr,
            logger=logger,
            prompt_hook=prompt_hook,
            annotation_prompt_mode=annotation_prompt_mode,
            include_annotation_samples=False,
        )

    loop = SelfRefineLoop(
        max_attempts=max_attempts,
        logger=logger,
        human_reviewer=human_reviewer,
        label=f"AnnoInstr-{calibration_source_label}",
    )
    result = loop.run(
        initial_artifact=initial_instruction,
        generate_fn=_generate_fn_wrapper,
        evaluate_fn=evaluate_fn,
        refine_fn=refine_fn,
    )

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for entry in result.history:
        _cache_anno_instruction_attempt(
            task, entry.attempt, entry.artifact, run_ts, logger
        )
    _cache_anno_instruction_refinement_history(task, result, run_ts, logger)

    instruction = result.artifact or initial_instruction
    _cache_anno_instruction(task, instruction, logger)
    _store_anno_instruction(task, instruction, logger)
    logger.info(
        "[SelfRefine-AnnoInstr-%s] Finished: passed=%s attempt=%d len=%d",
        calibration_source_label, result.passed, result.attempt, len(instruction),
    )
    return instruction


# ---------------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------------

def main():
    # Minimal logging until we have task and run_dir (file logging set up later)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)

    # ---- Pre-parse --config before building the full parser ----
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument("--config", type=str, default=None,
                             help="Path to config YAML file (default: config/config.yaml)")
    _pre_args, _ = _pre_parser.parse_known_args()
    cfg = load_config(_pre_args.config)
    configure_qwen_agent_openai_client_pool(cfg.get("llm_client_pool") or {}, logger)

    # ---- Pre-compute argparse defaults that need special handling ----
    _gpu_ids_default = _gpu_ids_to_str(cfg.get("index_gpu_ids", None))
    _q2p_gpu_ids_default = _gpu_ids_to_str(cfg.get("q2p_gpu_ids", None))
    _diverse_gpu_ids_default = _gpu_ids_to_str(cfg.get("diverse_selection_gpu_ids", None))

    # CLI: defaults from config, then CLI overrides
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Pipeline Main Controller")

    # ---- Config file ----
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file (default: config/config.yaml)")

    # ---- Task ----
    parser.add_argument("--task", type=str, default=cfg.get("task", "msmarco"),
                        help="Task name (e.g. msmarco, arguana, trec-covid)")

    # ---- Corpus & FAISS paths ----
    parser.add_argument("--corpus", type=str, default=cfg.get("corpus_path", ""),
                        dest="corpus_path", help="Path to corpus (TSV/JSONL/Parquet/HF)")
    parser.add_argument("--corpus-format", type=str, default=cfg.get("corpus_format", "auto"),
                        dest="corpus_format",
                        help="Corpus format: auto|tsv|csv|jsonl|parquet|hf_dataset")
    parser.add_argument("--faiss-dir", type=str, default=cfg.get("faiss_dir", ""),
                        dest="faiss_dir", help="Directory for FAISS index")

    # ---- Embedding model (shared by BuildIndex & Query2Passage) ----
    parser.add_argument("--embedding-model-name", type=str,
                        default=cfg.get("embedding_model_name", "BAAI/bge-m3"),
                        dest="embedding_model_name",
                        help="Sentence-transformer model name (e.g. BAAI/bge-m3)")
    parser.add_argument("--embedding-model-path", type=parse_optional_str,
                        default=parse_optional_str(cfg.get("embedding_model_path", None)),
                        dest="embedding_model_path",
                        help="Local path to embedding model directory; overrides --embedding-model-name when set")

    # ---- Step 2.1: BuildIndex specific ----
    parser.add_argument("--index-batch-size", type=int,
                        default=cfg.get("index_batch_size", 256),
                        dest="index_batch_size",
                        help="Batch size for corpus encoding during index building")
    parser.add_argument("--index-num-gpus", type=int,
                        default=cfg.get("index_num_gpus", 1),
                        dest="index_num_gpus",
                        help="Number of GPUs for parallel corpus encoding")
    parser.add_argument("--index-gpu-ids", type=str,
                        default=_gpu_ids_default,
                        dest="index_gpu_ids_str",
                        help="Comma-separated GPU IDs for encoding (e.g. 0,1,2,3); overrides --index-num-gpus")
    parser.add_argument("--no-index-faiss-gpu", action="store_true",
                        default=not cfg.get("index_use_faiss_gpu", True),
                        help="Disable GPU FAISS during index building")
    parser.add_argument("--no-index-faiss-fp16", action="store_true",
                        default=not cfg.get("index_faiss_gpu_float16", True),
                        help="Disable float16 for FAISS GPU during index building")
    parser.add_argument("--index-id-column", type=parse_optional_str,
                        default=parse_optional_str(cfg.get("index_id_column", None)),
                        dest="index_id_column",
                        help="Corpus ID column name (default: auto-detect)")
    parser.add_argument("--index-text-column", type=parse_optional_str,
                        default=parse_optional_str(cfg.get("index_text_column", None)),
                        dest="index_text_column",
                        help="Corpus text column name (default: auto-detect)")
    parser.add_argument("--index-title-column", type=parse_optional_str,
                        default=parse_optional_str(cfg.get("index_title_column", None)),
                        dest="index_title_column",
                        help='Corpus title column name (default: auto-detect; "" to disable)')

    # ---- Step 2.2: DiverseSelection specific ----
    parser.add_argument("--num-diverse-passages", type=parse_optional_int,
                        default=parse_optional_int(cfg.get("num_diverse_passages", None)),
                        dest="num_diverse_passages",
                        help="Number of diverse passages to select (target_num); use None for greedy selection (no limit)")
    parser.add_argument("--cosine-similarity-threshold", type=float,
                        default=cfg.get("cosine_similarity_threshold", 0.6),
                        dest="cosine_similarity_threshold",
                        help="Cosine similarity threshold for diverse selection")
    parser.add_argument("--diverse-selection-top-k", type=int,
                        default=cfg.get("top_k", 2000),
                        dest="diverse_selection_top_k",
                        help="Neighbour retrieval size per iteration in diverse selection")
    parser.add_argument("--diverse-selection-gpu-ids", type=str,
                        default=_diverse_gpu_ids_default,
                        dest="diverse_selection_gpu_ids_str",
                        help="Comma-separated GPU ids for IVF training in diverse selection "
                             "(only the first id is used; default: 0 when CUDA is available). "
                             "Use -1 to force CPU.")

    # ---- Step 3: Query generation ----
    _ablation_cfg = cfg.get("ablation") or {}
    parser.add_argument("--queries-per-passage", type=int,
                        default=cfg.get("queries_per_passage", 5),
                        dest="queries_per_passage",
                        help="Number of queries per passage")
    parser.add_argument("--idattr-resume-path", type=parse_optional_str,
                        default=parse_optional_str(cfg.get("idattr_resume_path", None)),
                        dest="idattr_resume_path",
                        help="Existing IdAttr JSONL to append to; already processed doc_ids are skipped")
    parser.add_argument("--disable-query-instruction", action=argparse.BooleanOptionalAction,
                        default=bool(_ablation_cfg.get("disable_query_instruction", False)),
                        dest="disable_query_instruction",
                        help="Ablation: skip refined query instruction generation/loading and pass empty "
                             "task_instruction to DiverseQuery/QueryFilter. Default: false.")
    parser.add_argument("--disable-idattr", action=argparse.BooleanOptionalAction,
                        default=bool(_ablation_cfg.get("disable_idattr", False)),
                        dest="disable_idattr",
                        help="Ablation: skip IdAttr generation and generate queries directly from documents. "
                             "Default: false.")

    # ---- Step 4: Query2Passage specific ----
    parser.add_argument("--top-k-candidates", type=int,
                        default=cfg.get("top_k_candidates", 20),
                        dest="top_k_candidates",
                        help="Number of candidates per query (Query2Passage)")
    parser.add_argument("--q2p-batch-size", type=int,
                        default=cfg.get("q2p_batch_size", 32),
                        dest="q2p_batch_size",
                        help="Batch size for query encoding in Query2Passage")
    parser.add_argument("--no-q2p-faiss-gpu", action="store_true",
                        default=not cfg.get("q2p_faiss_gpu", True),
                        help="Disable GPU FAISS during Query2Passage retrieval")
    parser.add_argument("--no-q2p-inject-origin-score", action="store_true",
                        default=not cfg.get("q2p_inject_origin_score", True),
                        dest="no_q2p_inject_origin_score",
                        help="Disable origin-doc cosine score injection in Query2Passage "
                             "(pos_score field written to each query object). Default: enabled.")
    parser.add_argument("--q2p-gpu-ids", type=str,
                        default=_q2p_gpu_ids_default,
                        dest="q2p_gpu_ids_str",
                        help="Comma-separated GPU ids for Query2Passage: FAISS shards across all listed "
                             "GPUs; embedding model uses the first id. Example: 4,5,6. "
                             "Omit or empty for default GPU 0.")
    parser.add_argument("--q2p-lock-path", type=str,
                        default=parse_optional_str(cfg.get("q2p_lock_path", None)),
                        dest="q2p_lock_path",
                        help="Optional file lock path for serializing Query2Passage across processes. "
                             "Default: disabled.")

    # ---- LLM (shared by IdAttr / QueryInstr / DiverseQuery / AnnoInstr / Annotation) ----
    parser.add_argument("--llm-model", type=str,
                        default=cfg.get("llm_model", "gpt-4o-mini"),
                        dest="llm_model",
                        help="LLM model name (e.g. gpt-4o-mini)")
    parser.add_argument("--llm-model-server", type=parse_optional_str,
                        default=parse_optional_str(cfg.get("llm_model_server", None)),
                        dest="llm_model_server",
                        help="OpenAI-compatible base URL; falls back to OPENAI_BASE_URL env var when not set")
    parser.add_argument("--llm-api-key", type=parse_optional_str,
                        default=parse_optional_str(cfg.get("llm_api_key", None)),
                        dest="llm_api_key",
                        help="LLM API key; falls back to OPENAI_API_KEY env var when not set")

    # ---- Step 5: Annotation specific (fully_annotation=true) ----
    parser.add_argument("--max-queries", type=parse_optional_int,
                        default=parse_optional_int(cfg.get("max_queries", None)),
                        dest="max_queries",
                        help="Max number of queries to annotate in batch annotation; None = annotate all")
    parser.add_argument("--candidates-num-each-anno", type=parse_optional_int,
                        default=parse_optional_int(cfg.get("candidates_num_each_anno", None)),
                        dest="candidates_num_each_anno",
                        help="Max candidates passed to the LLM per annotation call; "
                             "None = all candidates in one call (default). "
                             "Useful for reducing per-call token cost when top_k_candidates is large.")
    parser.add_argument("--annotation-prompt-mode",
                        choices=["multi_doc_with_doc_id", "single_doc_no_doc_id"],
                        default=cfg.get("annotation_prompt_mode", "multi_doc_with_doc_id"),
                        dest="annotation_prompt_mode",
                        help="Annotation prompt contract. multi_doc_with_doc_id is the existing schema; "
                             "single_doc_no_doc_id generates a single-doc instruction without doc_id IO "
                             "and forces one candidate per annotation call.")
    parser.add_argument("--hide-doc-id-for-single-doc", action=argparse.BooleanOptionalAction,
                        default=bool(cfg.get("hide_doc_id_for_single_doc", False)),
                        dest="hide_doc_id_for_single_doc",
                        help="Experimental annotation mode: when candidates_num_each_anno=1, "
                             "hide doc_id from the LLM and backfill the single input doc_id in code. "
                             "Useful for diagnosing DBpedia doc_id echo/matching failures. Default: false.")

    # ---- Step 5: Mode switch & HN mining (fully_annotation=false) ----
    parser.add_argument("--fully-annotation", action=argparse.BooleanOptionalAction,
                        default=cfg.get("fully_annotation", True),
                        dest="fully_annotation",
                        help="Step 5 mode: true=LLM full annotation (BatchCandidateAnnotator); "
                             "false=hard negative mining (HnMineTool). Default: true.")
    parser.add_argument("--origin-doc-annotation", action=argparse.BooleanOptionalAction,
                        default=cfg.get("origin_doc_annotation", False),
                        dest="origin_doc_annotation",
                        help="HN mode only: mine first, then LLM-verify original doc as positive; "
                             "writes HnMine_*_all.jsonl (full mined) and HnMine_*_origin_pos.jsonl "
                             "(filtered). Default: false.")
    parser.add_argument("--hn-top-k-candidates", type=int,
                        default=cfg.get("hn_top_k_candidates", 200),
                        dest="hn_top_k_candidates",
                        help="HN mode only: number of Q2P candidates to retrieve per query "
                             "(replaces --top-k-candidates in Step 4; Branch A needs 200+). Default: 200.")
    parser.add_argument("--hn-negative-number", type=int,
                        default=cfg.get("hn_negative_number", 15),
                        dest="hn_negative_number",
                        help="HN mode only: target number of hard negatives per query. Default: 15.")
    parser.add_argument("--hn-range-for-sampling", type=str,
                        default=cfg.get("hn_range_for_sampling", "30-200"),
                        dest="hn_range_for_sampling",
                        help="HN mode only: Branch A rank window, format 'start-end'. Default: '30-200'.")

    # ---- Cross-run instruction store ----
    parser.add_argument("--instructions-store-dir", type=parse_optional_str,
                        default=parse_optional_str(cfg.get("instructions_store_dir", None)),
                        dest="instructions_store_dir",
                        help="Directory to store/load final refined instructions across runs. "
                             "Default: Results/Instructions/{task_name}")

    # ---- Step 3.3: Query length distribution control ----
    _qlc_cfg = cfg.get("query_length_control", {}) or {}
    parser.add_argument("--query-length-control-enabled", action=argparse.BooleanOptionalAction,
                        default=bool(_qlc_cfg.get("enabled", False)),
                        dest="query_length_control_enabled",
                        help="Step 3.3: dynamically sample a query length bucket per document "
                             "and append a length constraint to task_instruction to align "
                             "synthesized query lengths with the golden distribution. "
                             "test mode generates + caches QueryLengthDist; "
                             "production mode loads the cache (skips if not found). "
                             "Default: False.")
    parser.add_argument("--query-length-control-use-llm", action=argparse.BooleanOptionalAction,
                        default=bool(_qlc_cfg.get("use_llm", False)),
                        dest="query_length_control_use_llm",
                        help="test mode only: use the LLM (queryinstr model) for semantically "
                             "richer bucket descriptions; ignored in production mode. "
                             "Default: False (fast statistical path).")

    # ---- Skip / test flags ----
    parser.add_argument("--filtered-diverse-texts", type=str,
                        default=cfg.get("filtered_diverse_texts"),
                        dest="filtered_diverse_texts",
                        help="Path to already-filtered diverse texts JSONL (skip Step 2 AND Step 2.5 if set)")
    parser.add_argument("--diverse-texts", type=str,
                        default=cfg.get("diverse_texts"),
                        dest="diverse_texts",
                        help="Path to diverse texts JSONL (skip Step 2 if set; Step 2.5 filter runs only when --use-filter)")
    parser.add_argument("--use-filter", action=argparse.BooleanOptionalAction,
                        default=cfg.get("use_filter", True),
                        dest="use_filter",
                        help="Run Step 2.5 DocType filter (default: True); use --no-use-filter to skip and pass "
                             "diverse texts directly to Step 3")
    parser.add_argument("--use-diverse-selection", action=argparse.BooleanOptionalAction,
                        default=cfg.get("use_diverse_selection", True),
                        dest="use_diverse_selection",
                        help="Step 2.2: run greedy diverse selection (default: True); "
                             "--no-use-diverse-selection exports all indexed texts as JSONL (same schema)")
    parser.add_argument("--filter-batch-size", type=int,
                        default=cfg.get("filter_batch_size", 10),
                        dest="filter_batch_size",
                        help="Number of docs sent to the LLM per call in Step 2.5 DocType filter (default: 10)")
    parser.add_argument("--filter-inject-pos-examples", action=argparse.BooleanOptionalAction,
                        default=cfg.get("filter_inject_pos_examples", False),
                        dest="filter_inject_pos_examples",
                        help="Step 2.5: inject few-shot positive examples into the filter prompt as "
                             "document-type prototypes (default: False)")
    parser.add_argument("--filter-pos-examples-count", type=int,
                        default=cfg.get("filter_pos_examples_count", 3),
                        dest="filter_pos_examples_count",
                        help="Maximum number of positive few-shot examples injected into the Step 2.5 "
                             "filter prompt (default: 3)")
    parser.add_argument("--filter-pos-example-max-chars", type=int,
                        default=cfg.get("filter_pos_example_max_chars", 1200),
                        dest="filter_pos_example_max_chars",
                        help="Maximum characters kept for each injected positive few-shot example in "
                             "Step 2.5 (default: 1200)")

    # ---- Step 3.5: Query filter ----
    parser.add_argument("--use-query-filter", action=argparse.BooleanOptionalAction,
                        default=cfg.get("use_query_filter", True),
                        dest="use_query_filter",
                        help="Run Step 3.5 Query filter (default: True); use --no-use-query-filter to skip "
                             "and pass DiverseQuery directly to Step 4")
    parser.add_argument("--query-filter-batch-size", type=int,
                        default=cfg.get("query_filter_batch_size", 4),
                        dest="query_filter_batch_size",
                        help="Number of queries sent to the LLM per call in Step 3.5 Query filter (default: 4)")
    parser.add_argument("--query-filter-inject-golden-queries", action=argparse.BooleanOptionalAction,
                        default=cfg.get("query_filter_inject_golden_queries", True),
                        dest="query_filter_inject_golden_queries",
                        help="Step 3.5: inject Few_Shot_Example queries as style benchmarks into the filter "
                             "prompt (default: True)")
    parser.add_argument("--query-filter-golden-query-count", type=int,
                        default=cfg.get("query_filter_golden_query_count", 3),
                        dest="query_filter_golden_query_count",
                        help="Maximum number of golden query examples injected into the Step 3.5 "
                             "filter prompt (default: 3)")
    parser.add_argument("--query-filter-golden-query-max-chars", type=int,
                        default=cfg.get("query_filter_golden_query_max_chars", 200),
                        dest="query_filter_golden_query_max_chars",
                        help="Maximum characters kept for each injected golden query example in "
                             "Step 3.5 (default: 200)")

    parser.add_argument("--skip-index", action="store_true",
                        default=cfg.get("skip_index", False),
                        help="Skip build_index. With --use-diverse-selection, requires existing index.faiss. "
                             "With --no-use-diverse-selection, id_map.pkl + doc_dict.pkl suffice for Step 2.2 export "
                             "(Step 4 still needs a full index).")
    parser.add_argument("--test-mode", action="store_true",
                        default=cfg.get("test_mode", False),
                        help="Run with limited data for testing (samples N diverse texts for Step 3)")
    parser.add_argument("--test-sample-size", type=int,
                        default=cfg.get("test_sample_size", 5),
                        dest="test_sample_size",
                        help="When --test-mode: number of diverse texts to sample for Step 3 (default: 5)")
    parser.add_argument("--test-seed", type=int,
                        default=cfg.get("test_seed", 42),
                        dest="test_seed",
                        help="When --test-mode: random seed for sampling (default: 42)")
    parser.add_argument("--human-review", action="store_true",
                        default=cfg.get("human_review", False),
                        dest="human_review",
                        help="Enable interactive human review after each self-refine evaluate step "
                             "(requires a TTY; only active in --test-mode)")
    parser.add_argument("--human-review-timeout", type=int,
                        default=cfg.get("human_review_timeout", 10),
                        dest="human_review_timeout",
                        help="Seconds to wait for the PASS override prompt before using the LLM decision (default: 10)")
    parser.add_argument("--human-review-no-editor", action="store_true",
                        default=cfg.get("human_review_no_editor", False),
                        dest="human_review_no_editor",
                        help="Disable opening $EDITOR for multi-line input; read from stdin instead")

    # ---- Annotation instruction self-refine calibration ----
    parser.add_argument("--anno-self-refine-source", type=str,
                        choices=["synthetic_q2p", "calibration_set", "fewshot_positive", "calibration_posneg"],
                        default=cfg.get("anno_self_refine_source", "synthetic_q2p"),
                        dest="anno_self_refine_source",
                        help="test-mode AnnoInstr self-refine source: legacy synthetic Q2P samples, "
                             "a confirmed calibration set, few-shot query-positive pairs only, or "
                             "a calibration set evaluated as positive vs. negative only")
    parser.add_argument("--anno-calibration-path", type=parse_optional_str,
                        default=parse_optional_str(cfg.get("anno_calibration_path", None)),
                        dest="anno_calibration_path",
                        help="Path to confirmed AnnoCalibration JSON used when "
                             "--anno-self-refine-source=calibration_set or calibration_posneg")
    parser.add_argument("--anno-fewshot-calibration-count", type=parse_optional_int,
                        default=parse_optional_int(cfg.get("anno_fewshot_calibration_count", None)),
                        dest="anno_fewshot_calibration_count",
                        help="Maximum few-shot query-positive examples used when "
                             "--anno-self-refine-source=fewshot_positive; None = all usable examples")
    parser.add_argument("--anno-self-refine-max-attempts", type=int,
                        default=int(cfg.get("anno_self_refine_max_attempts", 3)),
                        dest="anno_self_refine_max_attempts",
                        help="Maximum AnnoInstr self-refine attempts in test mode, including the initial attempt "
                             "(default: 3)")

    # ---- Production mode sampling (Step 2.5 → Step 3) ----
    _ps_cfg = cfg.get("production_sample") or {}
    parser.add_argument("--production-sample", action=argparse.BooleanOptionalAction,
                        default=bool(_ps_cfg.get("enabled", False)),
                        dest="production_sample_enabled",
                        help="Production mode only: sample a subset of diverse texts before Step 3 "
                             "(default: False). Has no effect in test mode.")
    parser.add_argument("--production-sample-mode", type=str,
                        default=_ps_cfg.get("mode", "random"),
                        choices=["random", "range", "indices"],
                        dest="production_sample_mode",
                        help="Sampling mode: random | range | indices (default: random)")
    parser.add_argument("--production-sample-count", type=parse_optional_int,
                        default=parse_optional_int(_ps_cfg.get("count", None)),
                        dest="production_sample_count",
                        help="mode=random: number of docs to sample")
    parser.add_argument("--production-sample-seed", type=parse_optional_int,
                        default=parse_optional_int(_ps_cfg.get("seed", None)),
                        dest="production_sample_seed",
                        help="mode=random: random seed for reproducibility (default: None = not fixed)")
    parser.add_argument("--production-sample-range", type=parse_optional_str,
                        default=parse_optional_str(_ps_cfg.get("range", None)),
                        dest="production_sample_range",
                        help="mode=range: 0-based closed index interval 'start-end', e.g. '0-99'")
    parser.add_argument("--production-sample-indices", type=parse_optional_str,
                        default=None,
                        dest="production_sample_indices_str",
                        help="mode=indices: comma-separated 0-based row indices, e.g. '0,5,10,42'")

    # ---- Step 4: cold start from DiverseQuery batch (skip Steps 2–3) ----
    parser.add_argument("--step4-diverse-query-input", type=parse_optional_str,
                        default=parse_optional_str(cfg.get("step4_diverse_query_input", None)),
                        dest="step4_diverse_query_input",
                        help="Path to DiverseQuery batch JSON (skip Steps 2, 2.5, 2.6, 3; enter at Step 4)")

    parser.add_argument("--step5-query2passage-input", type=parse_optional_str,
                        default=parse_optional_str(cfg.get("step5_query2passage_input", None)),
                        dest="step5_query2passage_input",
                        help="Path to Query2Passage output JSON (skip Steps 2–4; enter at Step 5; mutually exclusive with --step4-diverse-query-input)")

    args = parser.parse_args()

    if args.annotation_prompt_mode == "single_doc_no_doc_id":
        args.hide_doc_id_for_single_doc = True
        if args.candidates_num_each_anno != 1:
            logger.info(
                "annotation_prompt_mode=single_doc_no_doc_id requires candidates_num_each_anno=1; overriding %s -> 1",
                args.candidates_num_each_anno,
            )
            args.candidates_num_each_anno = 1

    # Periodic ``[Progress]`` lines in the main log (0 = disable heartbeats)
    progress_log_interval_sec = float(cfg.get("progress_log_interval_sec", 60.0))

    # ---- Post-parse: resolve concurrency config (YAML only; no CLI exposure needed) ----
    _conc_cfg = cfg.get("concurrency") or {}
    conc_enabled = bool(_conc_cfg.get("enabled", False))
    conc_max_workers = _conc_cfg.get("max_workers", None)  # None → Python default
    conc_jitter = float(_conc_cfg.get("jitter", 0.0))

    # ---- Post-parse: build production_sample config dict (merges YAML + CLI) ----
    # CLI always wins; fall back to YAML values for fields not explicitly overridden.
    _ps_indices_from_cli = None
    if args.production_sample_indices_str:
        _ps_indices_from_cli = [
            int(x.strip()) for x in args.production_sample_indices_str.split(",") if x.strip()
        ]
    production_sample_cfg = {
        "enabled": args.production_sample_enabled,
        "mode": args.production_sample_mode,
        "count": args.production_sample_count,
        "seed": args.production_sample_seed,
        "range": args.production_sample_range,
        "indices": _ps_indices_from_cli if _ps_indices_from_cli is not None else _ps_cfg.get("indices"),
    }

    # ---- Post-parse: build HumanReviewer (only meaningful in test-mode + TTY) ----
    human_reviewer: HumanReviewer | None = None
    if args.human_review and args.test_mode:
        human_reviewer = HumanReviewer(
            enabled=True,
            timeout=args.human_review_timeout,
            use_editor=not args.human_review_no_editor,
        )
        if not human_reviewer.enabled:
            # HumanReviewer auto-disables when stdin is not a TTY
            logger.warning(
                "--human-review requested but stdin is not a TTY; human review disabled."
            )
            human_reviewer = None
    elif args.human_review and not args.test_mode:
        logger.warning("--human-review has no effect without --test-mode; ignoring.")

    # ---- Post-parse: resolve GPU IDs ----
    index_gpu_ids = None
    if args.index_gpu_ids_str:
        index_gpu_ids = [int(x.strip()) for x in args.index_gpu_ids_str.split(",") if x.strip()]

    q2p_gpu_ids = None
    if args.q2p_gpu_ids_str:
        q2p_gpu_ids = [int(x.strip()) for x in args.q2p_gpu_ids_str.split(",") if x.strip()]

    # ---- Post-parse: resolve LLM config (config/CLI first, then env vars) ----
    llm_cfg = {
        "model": args.llm_model,
        "model_server": args.llm_model_server or os.getenv("OPENAI_BASE_URL"),
        "api_key": args.llm_api_key or os.getenv("OPENAI_API_KEY"),
    }
    # Per-stage overrides from config llm_stages
    # (idattr_refine | idattr_gen | queryinstr | diversequery | annoinstr | annotation | filter)
    # Backward compatibility: if idattr_refine/idattr_gen is absent but legacy idattr exists,
    # use legacy idattr for both refine and generation.
    stages_cfg = cfg.get("llm_stages") or {}
    llm_cfg_idattr_refine = get_llm_cfg_for_stage("idattr_refine", llm_cfg, cfg)
    llm_cfg_idattr_gen = get_llm_cfg_for_stage("idattr_gen", llm_cfg, cfg)
    llm_cfg_queryinstr = get_llm_cfg_for_stage("queryinstr", llm_cfg, cfg)
    llm_cfg_diversequery = get_llm_cfg_for_stage("diversequery", llm_cfg, cfg)
    llm_cfg_annoinstr = get_llm_cfg_for_stage("annoinstr", llm_cfg, cfg)
    llm_cfg_annotation = get_llm_cfg_for_stage("annotation", llm_cfg, cfg)
    llm_cfg_filter = get_llm_cfg_for_stage("filter", llm_cfg, cfg)
    llm_cfg_queryfilter = get_llm_cfg_for_stage("queryfilter", llm_cfg, cfg)

    # ---- Normalize task ----
    task = (args.task or "").strip().lower()
    if not task:
        logger.error("Task name is empty")
        return
    if task not in HighLevel_Task_Definition:
        logger.error("Task '%s' not found in HighLevel_Task_Definition. Valid: %s",
                     task, list(HighLevel_Task_Definition.keys()))
        return
    if task not in Few_Shot_Example:
        logger.error("Task '%s' not found in Few_Shot_Example. Valid: %s",
                     task, list(Few_Shot_Example.keys()))
        return

    # ---- Run directories: Logs/{task}_{timestamp}/, Results/{task}_{timestamp}/, Results/.../Instructions/ ----
    run_dir = f"{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir_run = f"Results/{run_dir}/Logs"
    results_dir_run = f"Results/{run_dir}"
    instructions_dir_run = f"Results/{run_dir}/Instructions"
    os.makedirs(log_dir_run, exist_ok=True)
    os.makedirs(results_dir_run, exist_ok=True)
    os.makedirs(instructions_dir_run, exist_ok=True)

    # Per-run refinement artifacts always go into this run's instructions dir.
    global _REFINE_CACHE_DIR, _INSTRUCTIONS_STORE_DIR
    _REFINE_CACHE_DIR = Path(__file__).resolve().parent / instructions_dir_run

    # Cross-run final instruction store: config/CLI > default Results/Instructions/{task}.
    store_dir_arg = args.instructions_store_dir or f"Results/Instructions/{task}"
    store_dir_path = Path(store_dir_arg)
    if not store_dir_path.is_absolute():
        store_dir_path = Path(__file__).resolve().parent / store_dir_path

    logger = setup_logging(log_dir_run)
    configure_qwen_agent_openai_client_pool(cfg.get("llm_client_pool") or {}, logger)

    _cfg_yaml = resolve_config_yaml_path(args.config)
    if _cfg_yaml.is_file():
        _cfg_dest = Path(__file__).resolve().parent / results_dir_run / _cfg_yaml.name
        shutil.copy2(_cfg_yaml, _cfg_dest)
        logger.info("Copied config YAML into run Results: %s (source=%s)", _cfg_dest, _cfg_yaml)
    else:
        logger.warning("Config YAML not found; skipped copy to Results: %s", _cfg_yaml)

    if args.test_mode:
        store_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        if not store_dir_path.is_dir():
            logger.error(
                "Production mode: instructions_store_dir does not exist: %s  "
                "(run test mode first, or set instructions_store_dir in config.yaml)",
                store_dir_path,
            )
            return

    _INSTRUCTIONS_STORE_DIR = store_dir_path

    # ---- Log run configuration summary ----
    run_config = {
        "task": task,
        # Corpus
        "corpus_path": args.corpus_path,
        "corpus_format": args.corpus_format,
        "faiss_dir": args.faiss_dir,
        # Embedding model
        "embedding_model_name": args.embedding_model_name,
        "embedding_model_path": args.embedding_model_path,
        # BuildIndex
        "index_batch_size": args.index_batch_size,
        "index_num_gpus": args.index_num_gpus,
        "index_gpu_ids": index_gpu_ids,
        "index_use_faiss_gpu": not args.no_index_faiss_gpu,
        "index_faiss_gpu_float16": not args.no_index_faiss_fp16,
        "index_id_column": args.index_id_column,
        "index_text_column": args.index_text_column,
        "index_title_column": args.index_title_column,
        # DiverseSelection
        "use_diverse_selection": args.use_diverse_selection,
        "num_diverse_passages": args.num_diverse_passages,
        "cosine_similarity_threshold": args.cosine_similarity_threshold,
        "diverse_selection_top_k": args.diverse_selection_top_k,
        # Queries
        "queries_per_passage": args.queries_per_passage,
        "idattr_resume_path": args.idattr_resume_path,
        "ablation": {
            "disable_query_instruction": args.disable_query_instruction,
            "disable_idattr": args.disable_idattr,
        },
        # Query2Passage
        "top_k_candidates": args.top_k_candidates,
        "q2p_batch_size": args.q2p_batch_size,
        "q2p_faiss_gpu": not args.no_q2p_faiss_gpu,
        "q2p_inject_origin_score": not args.no_q2p_inject_origin_score,
        "q2p_gpu_ids": q2p_gpu_ids,
        "q2p_lock_path": args.q2p_lock_path,
        # LLM
        "llm_model": llm_cfg["model"],
        "llm_model_server": llm_cfg["model_server"],
        # Step 5 mode
        "fully_annotation": args.fully_annotation,
        # Annotation (fully_annotation=true)
        "max_queries": args.max_queries,
        "candidates_num_each_anno": args.candidates_num_each_anno,
        "annotation_prompt_mode": args.annotation_prompt_mode,
        "hide_doc_id_for_single_doc": args.hide_doc_id_for_single_doc,
        # HN mining (fully_annotation=false)
        "origin_doc_annotation": args.origin_doc_annotation,
        "hn_top_k_candidates": args.hn_top_k_candidates,
        "hn_negative_number": args.hn_negative_number,
        "hn_range_for_sampling": args.hn_range_for_sampling,
        "hn_mine_output_path_all": None,
        "hn_mine_output_path_origin_filtered": None,
        # Skip / test
        "diverse_texts_arg": args.diverse_texts,
        "filter_inject_pos_examples": args.filter_inject_pos_examples,
        "filter_pos_examples_count": args.filter_pos_examples_count,
        "filter_pos_example_max_chars": args.filter_pos_example_max_chars,
        "skip_index": args.skip_index,
        "test_mode": args.test_mode,
        "refine_cache_dir": str(_REFINE_CACHE_DIR),
        "instructions_store_dir": str(_INSTRUCTIONS_STORE_DIR),
        # Production sampling
        "production_sample": production_sample_cfg,
        # Step 4 cold start
        "step4_diverse_query_input": parse_optional_str(args.step4_diverse_query_input),
        # Step 5 cold start (skip Steps 2–4)
        "step5_query2passage_input": parse_optional_str(args.step5_query2passage_input),
        # Concurrency
        "concurrency": {
            "enabled": conc_enabled,
            "max_workers": conc_max_workers,
            "jitter": conc_jitter,
        },
        "llm_client_pool": cfg.get("llm_client_pool") or {},
        "progress_log_interval_sec": progress_log_interval_sec,
        "anno_self_refine_source": args.anno_self_refine_source,
        "anno_calibration_path": args.anno_calibration_path,
        "anno_fewshot_calibration_count": args.anno_fewshot_calibration_count,
    }
    if args.test_mode:
        run_config["test_sample_size"] = args.test_sample_size
        run_config["test_seed"] = args.test_seed
        run_config["human_review"] = args.human_review
        if args.human_review:
            run_config["human_review_timeout"] = args.human_review_timeout
            run_config["human_review_no_editor"] = args.human_review_no_editor
    logger.info("=== Pipeline run config === %s", run_config)

    # ---------------------------------------------------------------------------
    #  Step 1: Task definition and few-shot (validation only)
    # ---------------------------------------------------------------------------
    logger.info("=== Step 1: Task definition and few-shot (loaded for %s) ===", task)
    t0_step1 = time.perf_counter()
    highlevel_task_definition = HighLevel_Task_Definition[task]
    few_shot_examples = Few_Shot_Example[task]
    logger.info("HighLevel task definition: %s", highlevel_task_definition)
    logger.info("Few-shot examples loaded (count: %d)",
                len(few_shot_examples) if isinstance(few_shot_examples, list) else 1)
    logger.info("=== Step 1 done === elapsed=%.2fs", time.perf_counter() - t0_step1)

    step4_src = parse_optional_str(args.step4_diverse_query_input)
    step5_src = parse_optional_str(args.step5_query2passage_input)
    if step5_src and step4_src:
        logger.error(
            "Cannot use both step5_query2passage_input and step4_diverse_query_input; "
            "set only one cold-start path."
        )
        return

    skip_step4_from_step5_input = False
    if step5_src:
        logger.info(
            "=== Steps 2, 2.5, 2.6, 3, 4: Skipped (step5_query2passage_input) === file=%s",
            step5_src,
        )
        q2p_output_path = materialize_step5_query2passage_input(step5_src, results_dir_run, logger)
        if q2p_output_path is None:
            return
        diverse_texts_path = "(skipped: step5_query2passage_input)"
        idattr_results_path = "(skipped: step5_query2passage_input)"
        dq_batch_path = "(skipped: step5_query2passage_input)"
        query_instruction = ""
        skip_step4_from_step5_input = True
    elif step4_src:
        if not str(args.faiss_dir or "").strip():
            logger.error("[Step4] faiss_dir must be set when using step4_diverse_query_input")
            return
        logger.info(
            "=== Step 2, 2.5, 2.6, 3: Skipped (step4_diverse_query_input) === file=%s",
            step4_src,
        )
        dq_batch_path = materialize_step4_diverse_query_input(step4_src, results_dir_run, logger)
        if dq_batch_path is None:
            return
        # step4_src is the DiverseQuery batch JSON, not diverse_texts JSONL — keep names honest for logs.
        diverse_texts_path = "(skipped: Steps 2–3; no diverse_texts JSONL)"
        idattr_results_path = "(skipped: Steps 2–3)"
        # Step 3.5 query filter may still run on the imported DQ batch; try to load a cached
        # query instruction so the style-guide tier of the filter prompt is populated.
        _cached_qi_for_filter = _load_cached_query_instruction(task, logger)
        query_instruction = _cached_qi_for_filter or ""
        if query_instruction:
            logger.info(
                "Step 4 cold-start: loaded cached query instruction for Step 3.5 filter (len=%d)",
                len(query_instruction),
            )
        else:
            logger.info(
                "Step 4 cold-start: no cached query instruction found; "
                "Step 3.5 filter will rely on golden queries only.",
            )
    else:
        # ---------------------------------------------------------------------------
        #  Step 2: Indexing and diverse selection (default: run; skip if --diverse-texts)
        #  Step 2.5: DocType filter (skip if --filtered-diverse-texts is provided)
        # ---------------------------------------------------------------------------
        filtered_diverse_texts_path = (
            getattr(args, "filtered_diverse_texts", None)
            and str(args.filtered_diverse_texts).strip()
            or None
        )
        diverse_texts_path = getattr(args, "diverse_texts", None) and str(args.diverse_texts).strip() or None
    
        if filtered_diverse_texts_path:
            # Both Step 2 and Step 2.5 are skipped; go directly to Step 3.
            logger.info(
                "=== Step 2 & 2.5: Skipped (using --filtered-diverse-texts) === input=%s",
                filtered_diverse_texts_path,
            )
            if not os.path.isfile(filtered_diverse_texts_path):
                logger.error("Filtered diverse texts file not found: %s", filtered_diverse_texts_path)
                return
            with open(filtered_diverse_texts_path, "r", encoding="utf-8") as f:
                n_lines = sum(1 for _ in f if _.strip())
            logger.info("Filtered diverse texts file lines: %d", n_lines)
            diverse_texts_path = filtered_diverse_texts_path
        else:
            # ---- Step 2: produce diverse_texts_path ----
            if diverse_texts_path:
                logger.info("=== Step 2: Skipped (using --diverse-texts) === input=%s", diverse_texts_path)
                if not os.path.isfile(diverse_texts_path):
                    logger.error("Diverse texts file not found: %s", diverse_texts_path)
                    return
                with open(diverse_texts_path, "r", encoding="utf-8") as f:
                    n_lines = sum(1 for _ in f if _.strip())
                logger.info("Diverse texts file lines: %d", n_lines)
            else:
                t0_step2 = time.perf_counter()
                logger.info(
                    "=== Step 2: Indexing and %s === faiss_dir=%s",
                    "diverse selection" if args.use_diverse_selection else "full text export",
                    args.faiss_dir,
                )

                if not args.skip_index:
                    logger.info("Step 2.1: Building FAISS index ... corpus_path=%s", args.corpus_path)
                    index_cfg = BuildIndexConfig(
                        corpus_path=args.corpus_path,
                        corpus_format=args.corpus_format,
                        faiss_index_dir=args.faiss_dir,
                        # Embedding model
                        model_name=args.embedding_model_name,
                        model_path=args.embedding_model_path,
                        # Batching & GPU
                        batch_size=args.index_batch_size,
                        num_gpus=args.index_num_gpus,
                        gpu_ids=index_gpu_ids,
                        use_faiss_gpu=not args.no_index_faiss_gpu,
                        faiss_gpu_float16=not args.no_index_faiss_fp16,
                        # Column mapping
                        id_column=args.index_id_column,
                        text_column=args.index_text_column,
                        title_column=args.index_title_column,
                    )
                    with periodic_progress_log(
                        logger, "Step 2.1 build_index (FAISS)", progress_log_interval_sec
                    ):
                        build_index(index_cfg, logger=logger)
                    logger.info("Step 2.1 done: index built at %s", args.faiss_dir)
                else:
                    logger.info("Step 2.1: Skipped (--skip-index); using existing artefacts at %s", args.faiss_dir)
                    if not args.use_diverse_selection:
                        logger.info(
                            "Step 2.2 will export from id_map.pkl + doc_dict.pkl only; "
                            "index.faiss is not required for that step (Step 4 Query2Passage still needs it)."
                        )

                if args.use_diverse_selection:
                    logger.info(
                        "Step 2.2: Selecting diverse texts ... cos_thresh=%s target_num=%s top_k=%s",
                        args.cosine_similarity_threshold,
                        args.num_diverse_passages,
                        args.diverse_selection_top_k,
                    )
                    _div_gpu_id = None
                    _div_device = None  # auto (cuda if available else cpu)
                    _div_s = args.diverse_selection_gpu_ids_str
                    if _div_s is not None and str(_div_s).strip():
                        _div_parts = [int(x.strip()) for x in str(_div_s).split(",") if x.strip()]
                        if _div_parts:
                            if _div_parts[0] < 0:
                                _div_device = "cpu"
                            else:
                                _div_gpu_id = _div_parts[0]
                                if len(_div_parts) > 1:
                                    logger.info(
                                        "diverse_selection: %d GPU ids in config; IVF training uses GPU %d only",
                                        len(_div_parts),
                                        _div_parts[0],
                                    )
                    div_cfg = DiverseSelectionConfig(
                        faiss_index_dir=args.faiss_dir,
                        cos_thresh=args.cosine_similarity_threshold,
                        top_k=args.diverse_selection_top_k,
                        target_num=args.num_diverse_passages,
                        output_dir=results_dir_run,
                        log_dir=log_dir_run,
                        gpu_id=_div_gpu_id,
                        device=_div_device,
                    )
                    with periodic_progress_log(
                        logger, "Step 2.2 diverse text selection", progress_log_interval_sec
                    ):
                        diverse_texts_path = select_diverse_texts(div_cfg, logger=logger)
                else:
                    logger.debug(
                        "Step 2.2: use_diverse_selection=false; ignoring cos_thresh, num_diverse_passages, "
                        "top_k, diverse_selection_gpu_ids"
                    )
                    logger.info("Step 2.2: Exporting all indexed texts (same JSONL schema as diverse selection)")
                    export_cfg = DiverseSelectionConfig(
                        faiss_index_dir=args.faiss_dir,
                        output_dir=results_dir_run,
                        log_dir=log_dir_run,
                        test_mode=args.test_mode,
                        test_select_limit=args.test_sample_size if args.test_mode else 20,
                    )
                    with periodic_progress_log(
                        logger, "Step 2.2 export all indexed texts", progress_log_interval_sec
                    ):
                        diverse_texts_path = export_all_indexed_texts(export_cfg, logger=logger)
                elapsed = time.perf_counter() - t0_step2
                with open(diverse_texts_path, "r", encoding="utf-8") as f:
                    n_diverse = sum(1 for _ in f if _.strip())
                logger.info("=== Step 2 done === output=%s lines=%d elapsed=%.2fs",
                            diverse_texts_path, n_diverse, elapsed)
    
            # ---- Step 2.5: DocType filter ----
            if not args.use_filter:
                logger.info(
                    "=== Step 2.5: Skipped (use_filter=false) === input=%s",
                    diverse_texts_path,
                )
            else:
                t0_step25 = time.perf_counter()
                logger.info(
                    "=== Step 2.5: DocType filter === input=%s task=%s batch_size=%d inject_pos_examples=%s",
                    diverse_texts_path, task, args.filter_batch_size, args.filter_inject_pos_examples,
                )
                filter_cfg = FilterConfig(
                    input_path=diverse_texts_path,
                    task_name=task,
                    output_dir=results_dir_run,
                    log_dir=log_dir_run,
                    batch_size=args.filter_batch_size,
                    model=llm_cfg_filter["model"],
                    model_server=llm_cfg_filter["model_server"],
                    api_key=llm_cfg_filter["api_key"],
                    concurrency_enabled=conc_enabled,
                    max_workers=conc_max_workers,
                    jitter=conc_jitter,
                    filter_inject_pos_examples=args.filter_inject_pos_examples,
                    filter_pos_examples_count=args.filter_pos_examples_count,
                    filter_pos_example_max_chars=args.filter_pos_example_max_chars,
                )
                with periodic_progress_log(
                    logger, "Step 2.5 DocType filter", progress_log_interval_sec
                ):
                    diverse_texts_path = filter_diverse_texts(filter_cfg, logger=logger)
                elapsed25 = time.perf_counter() - t0_step25
                with open(diverse_texts_path, "r", encoding="utf-8") as f:
                    n_filtered = sum(1 for _ in f if _.strip())
                logger.info(
                    "=== Step 2.5 done === output=%s lines=%d elapsed=%.2fs",
                    diverse_texts_path, n_filtered, elapsed25,
                )
    
        # ---------------------------------------------------------------------------
        #  Step 2.6 (optional): Production-mode sampling
        #  Samples a subset of the final diverse texts before Step 3 so that the
        #  expensive LLM stages only process the selected docs.
        #  The FAISS index (full corpus) used in Step 4 is NOT affected.
        # ---------------------------------------------------------------------------
        if not args.test_mode and production_sample_cfg.get("enabled"):
            t0_sample = time.perf_counter()
            logger.info(
                "=== Step 2.6: Production sampling === mode=%s input=%s",
                production_sample_cfg["mode"], diverse_texts_path,
            )
            with periodic_progress_log(
                logger, "Step 2.6 production sampling (load + sample)", progress_log_interval_sec
            ):
                with open(diverse_texts_path, "r", encoding="utf-8") as _f:
                    _all_docs = [json.loads(line) for line in _f if line.strip()]
                _sampled_docs = apply_production_sample(_all_docs, production_sample_cfg, logger)
            if len(_sampled_docs) == 0:
                logger.error(
                    "Production sampling returned 0 docs (mode=%s cfg=%s); aborting.",
                    production_sample_cfg["mode"], production_sample_cfg,
                )
                return
            _sample_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _sampled_path = (
                f"{results_dir_run}/sampled_diverse_texts_{production_sample_cfg['mode']}_{_sample_ts}.jsonl"
            )
            with open(_sampled_path, "w", encoding="utf-8") as _f:
                for _doc in _sampled_docs:
                    _f.write(json.dumps(_doc, ensure_ascii=False) + "\n")
            elapsed_sample = time.perf_counter() - t0_sample
            logger.info(
                "=== Step 2.6 done === total=%d sampled=%d output=%s elapsed=%.2fs",
                len(_all_docs), len(_sampled_docs), _sampled_path, elapsed_sample,
            )
            diverse_texts_path = _sampled_path
    
        # ---------------------------------------------------------------------------
        #  Step 3: IdAttr, Query instruction, Diverse queries
        # ---------------------------------------------------------------------------
        t0_step3 = time.perf_counter()
        _prod_sample_active = not args.test_mode and production_sample_cfg.get("enabled")
        if args.test_mode:
            logger.info(
                "=== Step 3: IdAttr + Query instruction + Diverse queries "
                "(TEST mode: sampling %d documents, self-refine enabled) "
                "=== input=%s task=%s",
                args.test_sample_size, diverse_texts_path, task,
            )
        elif _prod_sample_active:
            logger.info(
                "=== Step 3: IdAttr + Query instruction + Diverse queries "
                "(PRODUCTION mode: using sampled diverse texts, mode=%s) === input=%s task=%s",
                production_sample_cfg["mode"], diverse_texts_path, task,
            )
        else:
            logger.info(
                "=== Step 3: IdAttr + Query instruction + Diverse queries "
                "(PRODUCTION mode: using all diverse texts) === input=%s task=%s",
                diverse_texts_path, task,
            )
    
        # ---- Step 3.1: IdAttr generation (with self-refine in test-mode) --------
        idattr_limit = args.test_sample_size if args.test_mode else None
        idattr_seed = args.test_seed if args.test_mode else None
    
        if args.disable_idattr:
            logger.info(
                "Step 3.1 IdAttr skipped (ablation.disable_idattr=true); "
                "DiverseQuery will read documents directly from %s",
                diverse_texts_path,
            )
            refinement_guidance = ""
            idattr_results_path = "(skipped: ablation.disable_idattr=true)"
            n_idattr = 0
        elif args.test_mode:
            with periodic_progress_log(
                logger, "Step 3.0 IdAttr self-refine (test-mode)", progress_log_interval_sec
            ):
                refinement_guidance = _run_idattr_self_refine(
                    diverse_texts_path=diverse_texts_path,
                    task=task,
                    llm_cfg=llm_cfg_idattr_refine,
                    limit=idattr_limit,
                    seed=idattr_seed,
                    logger=logger,
                    human_reviewer=human_reviewer,
                    llm_cfg_gen=llm_cfg_idattr_gen,
                )
        else:
            refinement_guidance = _load_cached_idattr_refinement(task, logger)
            if refinement_guidance is None:
                logger.error(
                    "Production mode: no IdAttrRefine_%s_*.txt found in %s  "
                    "(run test mode first to generate/refine guidance; "
                    "empty guidance file is allowed)",
                    task, _INSTRUCTIONS_STORE_DIR,
                )
                return
    
        if not args.disable_idattr:
            idattr_cfg = IdAttrConfig(
                input_path=diverse_texts_path,
                task_name=task,
                output_dir=results_dir_run,
                log_dir=log_dir_run,
                model=llm_cfg_idattr_gen["model"],
                model_server=llm_cfg_idattr_gen["model_server"],
                api_key=llm_cfg_idattr_gen["api_key"],
                limit=idattr_limit,
                seed=idattr_seed,
                resume_path=args.idattr_resume_path,
                refinement_guidance=refinement_guidance,
                concurrency_enabled=conc_enabled,
                max_workers=conc_max_workers,
                jitter=conc_jitter,
                progress_log_interval_sec=progress_log_interval_sec,
            )
            idattr_results_path = generate_idattr(idattr_cfg, logger=logger)
            with open(idattr_results_path, "r", encoding="utf-8") as f:
                n_idattr = sum(1 for _ in f if _.strip())
        logger.info("Step 3.1 IdAttr done: output=%s lines=%d", idattr_results_path, n_idattr)
    
        # ---- Step 3.2 + 3.3: Query instruction + diverse queries ----------------
        if args.disable_query_instruction:
            query_instruction = ""
            logger.info(
                "Step 3.2 Query instruction skipped (ablation.disable_query_instruction=true); "
                "DiverseQuery and QueryFilter will receive an empty task_instruction."
            )
        elif args.test_mode:
            logger.info("Step 3.2+3.3: Query instruction self-refine (test-mode) ...")
            with periodic_progress_log(
                logger, "Step 3.2 QueryInstr self-refine (test-mode)", progress_log_interval_sec
            ):
                query_instruction = _run_query_instr_self_refine(
                    idattr_results_path=(
                        diverse_texts_path if args.disable_idattr else idattr_results_path
                    ),
                    task=task,
                    llm_cfg_queryinstr=llm_cfg_queryinstr,
                    llm_cfg_diversequery=llm_cfg_diversequery,
                    queries_per_passage=args.queries_per_passage,
                    logger=logger,
                    human_reviewer=human_reviewer,
                    use_idattr=not args.disable_idattr,
                    limit=idattr_limit if args.disable_idattr else None,
                    seed=idattr_seed if args.disable_idattr else None,
                )
            logger.info("Step 3.2: Query style guidelines generated (length=%d)", len(query_instruction))
        else:
            cached_qi = _load_cached_query_instruction(task, logger)
            if cached_qi:
                query_instruction = cached_qi
                logger.info("Step 3.2: Using cached refined query instruction (len=%d)",
                            len(query_instruction))
            else:
                logger.error(
                    "Production mode: no QueryInstrRefine_%s_*.txt found in %s  "
                    "(run test mode first to generate refined query instruction)",
                    task, _INSTRUCTIONS_STORE_DIR,
                )
                return
    
        # ---- Step 3.2b: Query length distribution sampler (optional) -----------
        # test mode  : generate (statistical or LLM), write to _REFINE_CACHE_DIR
        #              (per-run) AND _INSTRUCTIONS_STORE_DIR (cross-run store).
        # production : load from _INSTRUCTIONS_STORE_DIR only; warn + disable if missing.
        # --query-length-control-use-llm is a test-mode-only flag; ignored in production.
        length_sampler = None
        if args.query_length_control_enabled and args.disable_query_instruction:
            logger.info(
                "Step 3.2b Query length control skipped because "
                "ablation.disable_query_instruction=true."
            )
        elif args.query_length_control_enabled:
            if args.test_mode:
                if args.query_length_control_use_llm:
                    length_sampler = QueryLengthSampler.from_examples_with_llm(
                        task_name=task, llm_cfg=llm_cfg_queryinstr
                    )
                else:
                    length_sampler = QueryLengthSampler.from_examples_statistical(task)
                _cache_length_sampler(task, length_sampler, logger)
                _store_length_sampler(task, length_sampler, logger)
            else:
                length_sampler = _load_cached_length_sampler(task, logger)
                if length_sampler is None:
                    logger.warning(
                        "Production mode: no QueryLengthDist_%s_*.json found in %s; "
                        "query length control disabled for this run. "
                        "Run test mode first to generate the length distribution.",
                        task, _INSTRUCTIONS_STORE_DIR,
                    )
            if length_sampler is not None:
                logger.info(
                    "Step 3.2b: QueryLengthSampler ready (%s mode): %s",
                    "test" if args.test_mode else "production",
                    length_sampler,
                )

        logger.info(
            "Step 3.3: Generating diverse queries ... num_queries_per_passage=%d concurrency=%s max_workers=%s",
            args.queries_per_passage, conc_enabled, conc_max_workers,
        )
        dq_tool = DiverseQueryGenerator()
        dq_input_path = diverse_texts_path if args.disable_idattr else idattr_results_path
        with open(dq_input_path, "r", encoding="utf-8") as f:
            dq_input_lines = [line.strip() for line in f if line.strip()]
    
        def _generate_dq_for_one(it: dict) -> dict:
            doc_id = it.get("doc_id", "unknown")
            try:
                doc_text = get_doc_text_for_generation(it)
                effective_instruction = (
                    query_instruction + length_sampler.sample_length_hint()
                    if length_sampler is not None
                    else query_instruction
                )
                res_text = dq_tool.call(
                    json5.dumps({
                        "doc": doc_text,
                        "identifiers": it.get("identifiers", []),
                        "attributes": it.get("attributes", []),
                        "num_queries": args.queries_per_passage,
                        "task_instruction": effective_instruction,
                        "task_name": task,
                        "use_idattr": not args.disable_idattr,
                    }),
                    llm_cfg=llm_cfg_diversequery,
                )
                dq_result = json5.loads(res_text)
                dq_result["doc_id"] = doc_id
                dq_result["doc"] = doc_text
                return dq_result
            except Exception as exc:
                logger.error("DiverseQuery generation failed for doc_id=%s: %s", doc_id, exc)
                return {
                    "queries": [],
                    "doc_id": doc_id,
                    "doc": get_doc_text_for_generation(it),
                    "identifiers": [] if args.disable_idattr else it.get("identifiers", []),
                    "attributes": [] if args.disable_idattr else it.get("attributes", []),
                }
    
        id_attr_items = [json.loads(line) for line in dq_input_lines]
        if args.disable_idattr and idattr_limit is not None and idattr_limit < len(id_attr_items):
            import random as _random
            if idattr_seed is not None:
                _random.seed(idattr_seed)
            id_attr_items = _random.sample(id_attr_items, idattr_limit)
            logger.info(
                "Step 3.3 doc-only ablation sampled %d documents (seed=%s).",
                len(id_attr_items), idattr_seed,
            )
        logger.info(
            "Loaded %d %s items for DiverseQuery generation.",
            len(id_attr_items),
            "document" if args.disable_idattr else "IdAttr",
        )
        all_dq_results = run_concurrent(
            _generate_dq_for_one,
            [{"it": it} for it in id_attr_items],
            enabled=conc_enabled,
            max_workers=conc_max_workers,
            jitter=conc_jitter,
            progress_logger=logger,
            progress_label="Step 3.3 DiverseQuery",
            progress_interval_sec=progress_log_interval_sec,
        )
    
        dq_batch_path = f"{results_dir_run}/DiverseQuery_Main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(dq_batch_path, "w", encoding="utf-8") as f:
            json.dump({
                "input_path": dq_input_path,
                "num_queries": args.queries_per_passage,
                "use_idattr": not args.disable_idattr,
                "results": all_dq_results,
            }, f, ensure_ascii=False, indent=2)
        total_queries = sum(len(r.get("queries", [])) for r in all_dq_results)
        elapsed3 = time.perf_counter() - t0_step3
        logger.info("=== Step 3 done === idattr=%s dq_batch=%s total_queries=%d elapsed=%.2fs",
                    idattr_results_path, dq_batch_path, total_queries, elapsed3)

    # ---------------------------------------------------------------------------
    #  Step 3.5: Query Filter
    #  Filters per-query based on style/quality using a 3-tier LLM prompt:
    #  task context + query style guidelines + golden query examples.
    #  Updates dq_batch_path so Step 4 transparently reads the filtered file.
    #  Skipped when Step 4 itself is skipped (step5_query2passage_input cold-start).
    # ---------------------------------------------------------------------------
    if skip_step4_from_step5_input:
        logger.info("=== Step 3.5: Skipped (step5_query2passage_input cold-start) ===")
    elif not args.use_query_filter:
        logger.info(
            "=== Step 3.5: Skipped (use_query_filter=false) === dq_batch=%s",
            dq_batch_path,
        )
    else:
        t0_step35 = time.perf_counter()
        logger.info(
            "=== Step 3.5: Query filter === input=%s task=%s batch_size=%d "
            "inject_golden=%s golden_count=%d",
            dq_batch_path, task, args.query_filter_batch_size,
            args.query_filter_inject_golden_queries, args.query_filter_golden_query_count,
        )
        qf_config = QueryFilterConfig(
            input_path=dq_batch_path,
            task_name=task,
            query_instruction=query_instruction,
            output_dir=results_dir_run,
            log_dir=log_dir_run,
            batch_size=args.query_filter_batch_size,
            model=llm_cfg_queryfilter["model"],
            model_server=llm_cfg_queryfilter["model_server"],
            api_key=llm_cfg_queryfilter["api_key"],
            concurrency_enabled=conc_enabled,
            max_workers=conc_max_workers,
            jitter=conc_jitter,
            inject_golden_queries=args.query_filter_inject_golden_queries,
            golden_query_count=args.query_filter_golden_query_count,
            golden_query_max_chars=args.query_filter_golden_query_max_chars,
        )
        with periodic_progress_log(
            logger, "Step 3.5 Query filter", progress_log_interval_sec
        ):
            dq_batch_path = filter_diverse_queries(qf_config, logger=logger)
        elapsed35 = time.perf_counter() - t0_step35
        logger.info(
            "=== Step 3.5 done === filtered_dq=%s elapsed=%.2fs",
            dq_batch_path, elapsed35,
        )

    # ---------------------------------------------------------------------------
    #  Step 4: Query2Passage
    # ---------------------------------------------------------------------------
    t0_step4 = time.perf_counter()
    if skip_step4_from_step5_input:
        logger.info(
            "=== Step 4: Skipped (step5_query2passage_input) === using q2p_input=%s",
            q2p_output_path,
        )
        try:
            if file_is_large_q2p(q2p_output_path):
                n_q2p = count_queries_q2p_streaming(q2p_output_path)
            else:
                with open(q2p_output_path, "r", encoding="utf-8") as _f:
                    _q2p_loaded = json.load(_f)
                n_q2p = count_queries_in_q2p_json(_q2p_loaded)
        except OSError as exc:
            logger.error("[Step4] could not read Query2Passage input for stats: %s", exc)
            return
        except json.JSONDecodeError as exc:
            logger.error("[Step4] invalid JSON in Query2Passage input: %s", exc)
            return
        except Exception as exc:
            # e.g. ijson.common.IncompleteJSONError on truncated/corrupt files
            logger.error("[Step4] could not parse Query2Passage input for stats: %s", exc)
            return
        elapsed4 = time.perf_counter() - t0_step4
        logger.info(
            "=== Step 4 skipped === q2p_input=%s total_queries=%d elapsed=%.2fs",
            q2p_output_path,
            n_q2p,
            elapsed4,
        )
    else:
        # In HN mining mode retrieve more candidates so Branch A has a full rank window.
        q2p_top_k = args.top_k_candidates if args.fully_annotation else args.hn_top_k_candidates
        q2p_output_path = f"{results_dir_run}/Query2Passage_Main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        logger.info("=== Step 4: Query2Passage === query_file=%s faiss_dir=%s top_k=%d model=%s "
                    "(mode=%s)",
                    dq_batch_path, args.faiss_dir, q2p_top_k, args.embedding_model_name,
                    "full_annotation" if args.fully_annotation else "hn_mine")
        q2p_tool = Query2PassageTool()
        q2p_params = {
            "query_file": dq_batch_path,
            "faiss_dir": args.faiss_dir,
            "top_k": q2p_top_k,
            "output_file": q2p_output_path,
            "model_name": args.embedding_model_name,
            "batch_size": args.q2p_batch_size,
            "faiss_gpu": not args.no_q2p_faiss_gpu,
            "inject_origin_score": not args.no_q2p_inject_origin_score,
        }
        if args.embedding_model_path:
            q2p_params["model_path"] = args.embedding_model_path

        if q2p_gpu_ids:
            q2p_params["gpu_ids"] = q2p_gpu_ids
        with optional_file_lock(args.q2p_lock_path, logger, "Query2Passage"):
            q2p_result = q2p_tool.call(json5.dumps(q2p_params))

        q2p_result_dict = json5.loads(q2p_result)
        n_q2p = q2p_result_dict.get("total_queries", 0)
        elapsed4 = time.perf_counter() - t0_step4
        logger.info("=== Step 4 done === output=%s total_queries=%d elapsed=%.2fs",
                    q2p_output_path, n_q2p, elapsed4)

    # ---------------------------------------------------------------------------
    #  Step 5: Annotation instruction + Batch annotation  (fully_annotation=true)
    #       OR Hard negative mining                       (fully_annotation=false)
    # ---------------------------------------------------------------------------
    t0_step5 = time.perf_counter()
    logger.info("=== Step 5 === task=%s mode=%s",
                task, "full_annotation" if args.fully_annotation else "hn_mine")

    step5_output_path: str

    if args.fully_annotation:
        # ------------------------------------------------------------------
        # Branch A: LLM full annotation (original pipeline)
        # ------------------------------------------------------------------
        if args.test_mode:
            logger.info(
                "Step 5: Annotation instruction self-refine (test-mode, source=%s) ...",
                args.anno_self_refine_source,
            )
            with periodic_progress_log(
                logger, "Step 5.0 AnnoInstr self-refine (test-mode)", progress_log_interval_sec
            ):
                if args.anno_self_refine_source in {"calibration_set", "calibration_posneg"}:
                    if not args.anno_calibration_path:
                        logger.error(
                            "anno_self_refine_source=%s requires --anno-calibration-path "
                            "or anno_calibration_path in config.",
                            args.anno_self_refine_source,
                        )
                        return
                    is_posneg_calibration = args.anno_self_refine_source == "calibration_posneg"
                    annotation_instruction = _run_anno_instr_self_refine_from_calibration(
                        anno_calibration_path=args.anno_calibration_path,
                        task=task,
                        llm_cfg_annoinstr=llm_cfg_annoinstr,
                        llm_cfg_annotation=llm_cfg_annotation,
                        logger=logger,
                        human_reviewer=human_reviewer,
                        candidates_num_each_anno=args.candidates_num_each_anno,
                        hide_doc_id_for_single_doc=args.hide_doc_id_for_single_doc,
                        annotation_prompt_mode=args.annotation_prompt_mode,
                        max_attempts=args.anno_self_refine_max_attempts,
                        calibration_source_label=(
                            "CalibPosNeg" if is_posneg_calibration else "Calib"
                        ),
                        calibration_label_granularity=(
                            "posneg" if is_posneg_calibration else "three_way"
                        ),
                    )
                elif args.anno_self_refine_source == "fewshot_positive":
                    fewshot_calibration = build_positive_only_calibration_from_fewshot(
                        task,
                        golden_query_count=args.anno_fewshot_calibration_count,
                    )
                    annotation_instruction = _run_anno_instr_self_refine_from_calibration(
                        anno_calibration_path=None,
                        task=task,
                        llm_cfg_annoinstr=llm_cfg_annoinstr,
                        llm_cfg_annotation=llm_cfg_annotation,
                        logger=logger,
                        human_reviewer=human_reviewer,
                        candidates_num_each_anno=args.candidates_num_each_anno,
                        hide_doc_id_for_single_doc=args.hide_doc_id_for_single_doc,
                        annotation_prompt_mode=args.annotation_prompt_mode,
                        max_attempts=args.anno_self_refine_max_attempts,
                        calibration_data=fewshot_calibration,
                        calibration_source_label="FewShotPositive",
                    )
                else:
                    annotation_instruction = _run_anno_instr_self_refine(
                        q2p_output_path=q2p_output_path,
                        task=task,
                        llm_cfg_annoinstr=llm_cfg_annoinstr,
                        llm_cfg_annotation=llm_cfg_annotation,
                        logger=logger,
                        sample_size=args.test_sample_size,
                        human_reviewer=human_reviewer,
                        candidates_num_each_anno=args.candidates_num_each_anno,
                        hide_doc_id_for_single_doc=args.hide_doc_id_for_single_doc,
                        annotation_prompt_mode=args.annotation_prompt_mode,
                        max_attempts=args.anno_self_refine_max_attempts,
                    )
        else:
            cached_ai = _load_cached_anno_instruction(task, logger)
            if cached_ai:
                annotation_instruction = cached_ai
                logger.info("Step 5: Using cached refined annotation instruction (len=%d)",
                            len(annotation_instruction))
            else:
                logger.error(
                    "Production mode: no AnnoInstrRefine_%s_*.txt found in %s  "
                    "(run test mode first to generate refined annotation instruction)",
                    task, _INSTRUCTIONS_STORE_DIR,
                )
                return

        step5_output_path = f"{results_dir_run}/Annotated_Main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        annotator_tool = BatchCandidateAnnotator(
            task_name=task,
            llm_cfg=llm_cfg_annotation,
            instructions_dir=str(_INSTRUCTIONS_STORE_DIR),
            concurrency_enabled=conc_enabled,
            max_workers=conc_max_workers,
            jitter=conc_jitter,
            progress_logger=logger,
            progress_interval_sec=progress_log_interval_sec,
        )
        ann_params_dict = {
            "input_path": q2p_output_path,
            "output_path": step5_output_path,
            "task_name": task,
            "top_k": args.top_k_candidates,
            "annotation_prompt_mode": args.annotation_prompt_mode,
        }
        if args.max_queries is not None:
            ann_params_dict["max_queries"] = args.max_queries
        if args.candidates_num_each_anno is not None:
            ann_params_dict["candidates_num_each_anno"] = args.candidates_num_each_anno
        if args.hide_doc_id_for_single_doc:
            ann_params_dict["hide_doc_id_for_single_doc"] = True
        annotator_tool.call(
            json5.dumps(ann_params_dict, ensure_ascii=False), llm_cfg=llm_cfg_annotation
        )

    else:
        # ------------------------------------------------------------------
        # Branch B: Hard negative mining (HnMineTool)
        # ------------------------------------------------------------------
        # When origin_doc_annotation=True the annotation instruction is needed
        # to verify whether the original doc is a true positive.
        if args.origin_doc_annotation:
            if args.test_mode:
                logger.info(
                    "Step 5 (HN): Annotation instruction self-refine for origin_doc_annotation "
                    "(source=%s) ...",
                    args.anno_self_refine_source,
                )
                with periodic_progress_log(
                    logger, "Step 5.0 (HN) AnnoInstr self-refine (test-mode)", progress_log_interval_sec
                ):
                    if args.anno_self_refine_source == "calibration_set":
                        if not args.anno_calibration_path:
                            logger.error(
                                "anno_self_refine_source=calibration_set requires --anno-calibration-path "
                                "or anno_calibration_path in config."
                            )
                            return
                        _run_anno_instr_self_refine_from_calibration(
                            anno_calibration_path=args.anno_calibration_path,
                            task=task,
                            llm_cfg_annoinstr=llm_cfg_annoinstr,
                            llm_cfg_annotation=llm_cfg_annotation,
                            logger=logger,
                            human_reviewer=human_reviewer,
                            max_attempts=args.anno_self_refine_max_attempts,
                        )
                    else:
                        _run_anno_instr_self_refine(
                            q2p_output_path=q2p_output_path,
                            task=task,
                            llm_cfg_annoinstr=llm_cfg_annoinstr,
                            llm_cfg_annotation=llm_cfg_annotation,
                            logger=logger,
                            sample_size=args.test_sample_size,
                            human_reviewer=human_reviewer,
                            max_attempts=args.anno_self_refine_max_attempts,
                        )
            else:
                cached_ai = _load_cached_anno_instruction(task, logger)
                if not cached_ai:
                    logger.error(
                        "Production mode (HN + origin_doc_annotation): "
                        "no AnnoInstrRefine_%s_*.txt found in %s  "
                        "(run test mode first to generate refined annotation instruction)",
                        task, _INSTRUCTIONS_STORE_DIR,
                    )
                    return
                logger.info("Step 5 (HN): Loaded cached annotation instruction for origin_doc_annotation (len=%d)",
                            len(cached_ai))
        else:
            logger.info("Step 5 (HN): origin_doc_annotation=false — skipping annotation instruction.")

        _hn_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.origin_doc_annotation:
            hn_path_all = f"{results_dir_run}/HnMine_Main_{_hn_ts}_all.jsonl"
            hn_path_origin_pos = f"{results_dir_run}/HnMine_Main_{_hn_ts}_origin_pos.jsonl"
            step5_output_path = hn_path_origin_pos
        else:
            hn_path_all = f"{results_dir_run}/HnMine_Main_{_hn_ts}_all.jsonl"
            hn_path_origin_pos = None
            step5_output_path = hn_path_all

        hn_tool = HnMineTool(
            task_name=task,
            llm_cfg=llm_cfg_annotation,
            instructions_dir=str(_INSTRUCTIONS_STORE_DIR),
            concurrency_enabled=conc_enabled,
            max_workers=conc_max_workers,
            jitter=conc_jitter,
            progress_logger=logger,
            progress_interval_sec=progress_log_interval_sec,
        )
        hn_params_dict = {
            "input_path": q2p_output_path,
            "output_path": hn_path_all,
            "task_name": task,
            "origin_doc_annotation": args.origin_doc_annotation,
            "negative_number": args.hn_negative_number,
            "range_for_sampling": args.hn_range_for_sampling,
        }
        if hn_path_origin_pos is not None:
            hn_params_dict["output_path_origin_filtered"] = hn_path_origin_pos
        if args.max_queries is not None:
            hn_params_dict["max_queries"] = args.max_queries
        hn_result = hn_tool.call(
            json5.dumps(hn_params_dict, ensure_ascii=False), llm_cfg=llm_cfg_annotation
        )
        hn_result_dict = json5.loads(hn_result)
        run_config["hn_mine_output_path_all"] = hn_result_dict.get("output_path")
        run_config["hn_mine_output_path_origin_filtered"] = hn_result_dict.get(
            "output_path_origin_filtered"
        )
        logger.info("Step 5 (HN) stats: %s", hn_result_dict.get("stats", {}))
        if args.origin_doc_annotation:
            logger.info(
                "Step 5 (HN) dual outputs: all_mined=%s origin_pos=%s",
                hn_result_dict.get("output_path"),
                hn_result_dict.get("output_path_origin_filtered"),
            )

    elapsed5 = time.perf_counter() - t0_step5
    logger.info("=== Step 5 done === output=%s elapsed=%.2fs", step5_output_path, elapsed5)

    if (
        not args.fully_annotation
        and args.origin_doc_annotation
        and run_config.get("hn_mine_output_path_all")
    ):
        logger.info(
            "Pipeline run outputs: diverse_texts=%s | idattr=%s | dq=%s | q2p=%s | "
            "step5=%s | hn_mine_all=%s",
            diverse_texts_path,
            idattr_results_path,
            dq_batch_path,
            q2p_output_path,
            step5_output_path,
            run_config["hn_mine_output_path_all"],
        )
    else:
        logger.info(
            "Pipeline run outputs: diverse_texts=%s | idattr=%s | dq=%s | q2p=%s | step5=%s",
            diverse_texts_path, idattr_results_path, dq_batch_path, q2p_output_path, step5_output_path,
        )
    logger.info("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
