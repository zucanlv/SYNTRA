"""
python Batch_Candidate_Annotator.py \
  --input Results/Query2Passage_DiverseQuery_Batch_20260203_043528_592874_one.json \
  --output Results/Annotated_DiverseQuery_Batch_20260203_043528_592874_one.json \
  --task msmarco 
"""


import os
import json
import json5
import logging
from pathlib import Path
from typing import Any, List, Dict, Optional
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from datetime import datetime
from tqdm import tqdm
import argparse

from concurrent_runner import run_concurrent
from q2p_json_stream import LARGE_Q2P_BYTES

# Directory where AnnoInstr_Generator saves instructions (assume it has already been run).
INSTRUCTIONS_DIR = Path(__file__).resolve().parent / "Results" / "Instructions"

# The pipeline entry point owns logging when this module is imported.
if __name__ == "__main__":
    os.makedirs("Logs", exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"Logs/Batch_Candidate_Annotator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

@register_tool('batch_candidate_annotator')
class BatchCandidateAnnotator(BaseTool):
    """Annotate (query, candidate_documents): single call or batch file. Only entry is call(params)."""
    description = "Annotate candidate documents for a query (score 0-3, positive/hard negative/easy negative). Pass (query, candidates) for one query, or (input_path, output_path) to process a Query2Passage JSON file."
    parameters = [
        {'name': 'query', 'type': 'string', 'description': 'The query (for single-call mode).', 'required': False},
        {'name': 'candidates', 'type': 'list', 'description': 'Candidate documents with doc_id and text (for single-call mode).', 'required': False},
        {'name': 'input_path', 'type': 'string', 'description': 'Path to Query2Passage JSON file (batch mode).', 'required': False},
        {'name': 'output_path', 'type': 'string', 'description': 'Output path for batch mode. Default: Results/Annotated_<input_basename>.', 'required': False},
        {'name': 'task_name', 'type': 'string', 'description': 'Task name (e.g. MSMarco).', 'required': False},
        {'name': 'max_queries', 'type': 'number', 'description': 'Max queries to process in batch mode.', 'required': False},
        {'name': 'top_k', 'type': 'number', 'description': 'Top K candidates per query in batch mode.', 'required': False},
        {'name': 'candidates_num_each_anno', 'type': 'number',
         'description': 'Max candidates passed to the LLM in a single annotation call. '
                        'When a query has more candidates than this, they are split into chunks '
                        'and annotated sequentially; results are merged automatically. '
                        'None (default) = all candidates in one call (original behaviour).',
         'required': False},
        {'name': 'annotation_prompt_mode', 'type': 'string',
         'description': 'Annotation prompt contract: multi_doc_with_doc_id or single_doc_no_doc_id. '
                        'single_doc_no_doc_id implies hide_doc_id_for_single_doc when chunks contain one candidate.',
         'required': False},
        {'name': 'hide_doc_id_for_single_doc', 'type': 'boolean',
         'description': 'Experimental mode: when each annotation chunk has exactly one candidate, '
                        'do not show doc_id to the LLM and backfill the input doc_id in code. '
                        'Use with candidates_num_each_anno=1 to diagnose doc_id echo failures.',
         'required': False},
    ]

    def __init__(
        self,
        task_name: str = "msmarco",
        llm_cfg: Dict = None,
        instructions_dir: Optional[str] = None,
        concurrency_enabled: bool = False,
        max_workers: Optional[int] = None,
        jitter: float = 0.0,
        progress_logger: Optional[Any] = None,
        progress_interval_sec: float = 0.0,
    ):
        self.task_name = task_name
        self.llm_cfg = llm_cfg or {
            'model': 'gpt-4o-mini',
            'model_server': os.getenv('OPENAI_BASE_URL'),
            'api_key': os.getenv('OPENAI_API_KEY'),
        }
        self.instructions_dir = Path(instructions_dir) if instructions_dir else INSTRUCTIONS_DIR
        self.evaluation_instruction = None
        self.annotator = None
        self.concurrency_enabled = concurrency_enabled
        self.max_workers = max_workers
        self.jitter = jitter
        self.progress_logger = progress_logger
        self.progress_interval_sec = progress_interval_sec
        self.stats = {
            'total_queries': 0, 'total_passages': 0,
            'positive_passages': 0, 'hard_negative_passages': 0, 'easy_negative_passages': 0,
            'parse_errors': 0, 'retries': 0
        }

    def _load_instruction_for_task(self, task_name: str) -> Optional[str]:
        """Load annotation instruction from instructions dir (Results/Instructions or run-specific).

        Prefers refined instruction: AnnoInstrRefine_{task}_*.txt (excluding _attempt).
        Falls back to AnnoInstr_{task}_*.txt from AnnoInstr_Generator.
        """
        instr_dir = self.instructions_dir
        if not instr_dir.exists():
            logging.error(f"Instructions directory not found: {instr_dir}. Run AnnoInstr_Generator first.")
            return None
        # Prefer refined instruction (same file main uses; no duplicate write)
        refine_candidates = [p for p in instr_dir.glob(f"AnnoInstrRefine_{task_name}_*.txt") if "_attempt" not in p.name]
        if refine_candidates:
            refine_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            path = refine_candidates[0]
        else:
            candidates = list(instr_dir.glob(f"AnnoInstr_{task_name}_*.txt"))
            if not candidates:
                logging.error(f"No instruction file for task '{task_name}' in {instr_dir}. Run AnnoInstr_Generator first.")
                return None
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            path = candidates[0]
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        logging.info(f"Loaded annotation instruction from {path.name}")
        return content

    def _ensure_annotator(self, task_name: str, llm_cfg: Dict) -> None:
        """Load the annotation instruction and store llm_cfg for later per-thread use.

        We deliberately do NOT create a single shared ``self.annotator`` here so
        that concurrent threads can each create their own ``Assistant`` instance in
        ``_annotate_chunk`` without sharing mutable state.
        """
        if self.evaluation_instruction is not None and self.task_name == task_name and self.llm_cfg == llm_cfg:
            return
        self.task_name = task_name
        self.llm_cfg = llm_cfg
        logging.info(f"Loading annotation instruction for task '{task_name}' from {self.instructions_dir}...")
        self.evaluation_instruction = self._load_instruction_for_task(task_name)
        if not self.evaluation_instruction:
            raise FileNotFoundError(
                f"Annotation instruction for task '{task_name}' not found under {self.instructions_dir}. "
                "Run AnnoInstr_Generator.py first to generate instructions."
            )
        logging.info("Annotation instruction ready.")

    def call(self, params: str, **kwargs) -> str:
        params_dict = json5.loads(params)
        task_name = params_dict.get('task_name', self.task_name)
        llm_cfg = kwargs.get('llm_cfg', self.llm_cfg)
        input_path = params_dict.get('input_path') or params_dict.get('input')
        output_path = params_dict.get('output_path') or params_dict.get('output')
        max_queries = params_dict.get('max_queries')
        top_k = params_dict.get('top_k')
        raw_cna = params_dict.get('candidates_num_each_anno')
        candidates_num_each_anno = int(raw_cna) if raw_cna is not None and int(raw_cna) > 0 else None
        annotation_prompt_mode = str(params_dict.get('annotation_prompt_mode') or 'multi_doc_with_doc_id').strip()
        if annotation_prompt_mode == 'single_doc_no_doc_id':
            candidates_num_each_anno = 1
        hide_doc_id_for_single_doc = (
            bool(params_dict.get('hide_doc_id_for_single_doc', False))
            or annotation_prompt_mode == 'single_doc_no_doc_id'
        )

        # Batch mode: process file
        if input_path and isinstance(input_path, str) and input_path.strip():
            if not os.path.exists(input_path):
                return json5.dumps({'error': f'Input file not found: {input_path}'}, ensure_ascii=False)
            try:
                if os.path.getsize(input_path) >= LARGE_Q2P_BYTES:
                    return json5.dumps(
                        {
                            "error": (
                                f"Input file is too large ({os.path.getsize(input_path)} bytes >= {LARGE_Q2P_BYTES}) "
                                "for in-memory batch annotation (json.load loads the entire file). "
                                "Use hard-negative mining mode, split the JSON, or run on a machine with enough RAM."
                            )
                        },
                        ensure_ascii=False,
                    )
            except OSError as exc:
                return json5.dumps({"error": f"Cannot stat input file: {exc}"}, ensure_ascii=False)
            if not output_path or not str(output_path).strip():
                output_path = f"Results/Annotated_{os.path.basename(input_path)}"
            self._ensure_annotator(task_name, llm_cfg)
            self.stats = {
                'total_queries': 0, 'total_passages': 0,
                'positive_passages': 0, 'hard_negative_passages': 0, 'easy_negative_passages': 0,
                'parse_errors': 0, 'retries': 0
            }
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results = data.get('results', [])

            # Collect all (q_obj, candidates) pairs upfront so we can run concurrently.
            # Each entry is (q_obj_ref, truncated_candidates) where q_obj_ref is the
            # actual dict inside `data` — results are written back by reference after
            # concurrent execution.
            query_work: List[tuple] = []
            for item in results:
                for q_obj in item.get('queries', []):
                    if max_queries is not None and len(query_work) >= int(max_queries):
                        break
                    if not isinstance(q_obj, dict):
                        continue
                    query_text = q_obj.get('query', '')
                    candidates = q_obj.get('candidates', [])
                    if top_k is not None:
                        candidates = candidates[: int(top_k)]
                    if not query_text or not candidates:
                        continue
                    query_work.append((q_obj, query_text, candidates))
                if max_queries is not None and len(query_work) >= int(max_queries):
                    break

            total_to_process = len(query_work)
            logging.info(
                "Starting file processing: %d queries (concurrency=%s, max_workers=%s)",
                total_to_process, self.concurrency_enabled, self.max_workers,
            )

            def _annotate_query(query_text: str, candidates: list) -> Optional[Dict]:
                return self._annotate_one(
                    query_text, candidates,
                    candidates_num_each_anno=candidates_num_each_anno,
                    hide_doc_id_for_single_doc=hide_doc_id_for_single_doc,
                )

            annotations = run_concurrent(
                _annotate_query,
                [{"query_text": qt, "candidates": cands}
                 for _q_obj, qt, cands in query_work],
                enabled=self.concurrency_enabled,
                max_workers=self.max_workers,
                jitter=self.jitter,
                progress_logger=self.progress_logger,
                progress_label="Step 5 batch candidate annotation (LLM)",
                progress_interval_sec=self.progress_interval_sec,
            )

            pbar = tqdm(total=total_to_process, desc="Annotation progress")
            for (q_obj, _qt, candidates), annotation in zip(query_work, annotations):
                if annotation:
                    q_obj['annotations'] = annotation.get('annotations', [])
                    q_obj['positive_doc_ids'] = annotation.get('positive_doc_ids', [])
                    q_obj['hard_negative_doc_ids'] = annotation.get('hard_negative_doc_ids', [])
                    q_obj['easy_negative_doc_ids'] = annotation.get('easy_negative_doc_ids', [])
                    self.stats['total_queries'] += 1
                    self.stats['total_passages'] += len(candidates)
                    self.stats['positive_passages'] += len(q_obj['positive_doc_ids'])
                    self.stats['hard_negative_passages'] += len(q_obj['hard_negative_doc_ids'])
                    self.stats['easy_negative_passages'] += len(q_obj['easy_negative_doc_ids'])
                pbar.update(1)
            pbar.close()
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(f"Annotation complete. Results saved to: {output_path}. Stats: {self.stats}")
            return json5.dumps({'output_path': output_path, 'stats': self.stats}, ensure_ascii=False)

        return json5.dumps({'error': 'Missing input_path. Provide input_path (or input) for batch annotation.'}, ensure_ascii=False)

    def _extract_json(self, text: str) -> Dict:
        try:
            if '```json' in text:
                json_start = text.find('```json') + 7
                json_end = text.find('```', json_start)
                text = text[json_start:json_end].strip()
            elif '```' in text:
                json_start = text.find('```') + 3
                json_end = text.find('```', json_start)
                text = text[json_start:json_end].strip()
            text = text.strip().lstrip('`').rstrip('`')
            return json.loads(text)
        except Exception as e:
            logging.debug(f"JSON extraction failed: {e}")
            return None

    def _annotate_one(
        self,
        query: str,
        candidates: List[Dict],
        max_retries: int = 1,
        candidates_num_each_anno: Optional[int] = None,
        hide_doc_id_for_single_doc: bool = False,
    ) -> Optional[Dict]:
        """Annotate all candidates for one query.

        When candidates_num_each_anno is set and smaller than len(candidates),
        the candidates are split into chunks and each chunk is sent to the LLM
        in a separate call; results are merged transparently.
        """
        if not candidates_num_each_anno or len(candidates) <= candidates_num_each_anno:
            return self._annotate_chunk(
                query, candidates, max_retries,
                hide_doc_id_for_single_doc=hide_doc_id_for_single_doc,
            )

        merged: Dict[str, Any] = {
            'annotations': [],
            'positive_doc_ids': [],
            'hard_negative_doc_ids': [],
            'easy_negative_doc_ids': [],
        }
        any_success = False
        for chunk_start in range(0, len(candidates), candidates_num_each_anno):
            chunk = candidates[chunk_start: chunk_start + candidates_num_each_anno]
            result = self._annotate_chunk(
                query, chunk, max_retries,
                hide_doc_id_for_single_doc=hide_doc_id_for_single_doc,
            )
            if result:
                any_success = True
                merged['annotations'].extend(result.get('annotations', []))
                merged['positive_doc_ids'].extend(result.get('positive_doc_ids', []))
                merged['hard_negative_doc_ids'].extend(result.get('hard_negative_doc_ids', []))
                merged['easy_negative_doc_ids'].extend(result.get('easy_negative_doc_ids', []))
        return merged if any_success else None

    def _annotation_from_single_doc_output(self, annotation: Dict, doc_id: str) -> Dict:
        """Convert a single-doc, no-doc_id LLM output into the standard schema."""
        if isinstance(annotation.get('annotations'), list) and annotation['annotations']:
            source = annotation['annotations'][0]
        else:
            source = annotation
        item = {
            'doc_id': str(doc_id),
            'score': source.get('score'),
            'classification': source.get('classification'),
            'reasoning': source.get('reasoning', ''),
        }
        return {'annotations': [item]}

    def _annotate_chunk(
        self,
        query: str,
        candidates: List[Dict],
        max_retries: int = 1,
        hide_doc_id_for_single_doc: bool = False,
    ) -> Optional[Dict]:
        """Send one LLM call for a single (query, candidates-chunk) pair and parse the result.

        A fresh ``Assistant`` is created per call so that multiple threads can
        invoke this method concurrently without sharing mutable state.
        """
        single_doc_hidden_id = hide_doc_id_for_single_doc and len(candidates) == 1
        local_to_original_doc_id: Dict[str, str] = {}
        if single_doc_hidden_id:
            user_input = (
                f"Query: {query}\n\n"
                "Candidate document:\n\n"
                f"{candidates[0].get('doc', '')}\n\n"
                "Classify this single document as exactly one of: positive, hard_negative, easy_negative. "
                "Output ONLY a valid JSON object with keys: "
                "score, classification, reasoning."
            )
        else:
            user_input = f"Query: {query}\n\nCandidate documents (each with doc_id and doc):\n\n"
        input_doc_ids = set()
        for idx, c in enumerate(candidates):
            original_doc_id = str(c.get('doc_id', 'unknown'))
            if not single_doc_hidden_id:
                doc_id = str(idx)
                local_to_original_doc_id[doc_id] = original_doc_id
                input_doc_ids.add(doc_id)
                doc = c.get('doc', '')
                user_input += f"[doc_id: {doc_id}]\n{doc}\n\n"
            else:
                input_doc_ids.add(original_doc_id)
        messages = [{'role': 'user', 'content': user_input}]
        system_message = self.evaluation_instruction
        if single_doc_hidden_id:
            system_message = (
                f"{system_message}\n\n"
                "Experiment override for this call: there is exactly one candidate document. "
                "The user message intentionally omits doc_id. Return only score, "
                "classification, and reasoning; the caller will attach the doc_id."
            )
        annotator = Assistant(llm=self.llm_cfg, system_message=system_message)
        for attempt in range(max_retries + 1):
            try:
                result_text = ""
                for response in annotator.run(messages):
                    result_text = response[-1]['content']
                annotation = self._extract_json(result_text)
                if annotation:
                    if single_doc_hidden_id:
                        annotation = self._annotation_from_single_doc_output(
                            annotation, candidates[0].get('doc_id', 'unknown')
                        )
                    output_annotations = annotation.get('annotations', [])
                    valid_annotations = [a for a in output_annotations if str(a.get('doc_id')) in input_doc_ids]
                    if not single_doc_hidden_id:
                        for a in valid_annotations:
                            local_doc_id = str(a.get('doc_id'))
                            a['doc_id'] = local_to_original_doc_id[local_doc_id]
                    annotation['annotations'] = valid_annotations

                    def _norm_cls(c):
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

                    annotation['positive_doc_ids'] = [str(a['doc_id']) for a in valid_annotations if _norm_cls(a.get('classification')) == 'positive']
                    annotation['hard_negative_doc_ids'] = [str(a['doc_id']) for a in valid_annotations if _norm_cls(a.get('classification')) == 'hard_negative']
                    annotation['easy_negative_doc_ids'] = [str(a['doc_id']) for a in valid_annotations if _norm_cls(a.get('classification')) == 'easy_negative']
                    return annotation
                if attempt < max_retries:
                    logging.warning(f"Parse failed, retrying (attempt {attempt + 1})...")
                    messages.append({'role': 'assistant', 'content': result_text})
                    messages.append({'role': 'user', 'content': "Your output was not a valid JSON or was missing required fields. Please output ONLY the strictly valid JSON object with annotations, positive_pids, hard_negative_pids, and easy_negative_pids as specified in the system instructions."})
                    self.stats['retries'] += 1
            except Exception as e:
                logging.error(f"Error during annotation: {e}")
                if attempt >= max_retries:
                    break
        self.stats['parse_errors'] += 1
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch score and annotate Query2Passage results (positive / hard negative / easy negative classification).")
    parser.add_argument("--input", type=str, required=True, help="Input Query2Passage JSON file path")
    parser.add_argument("--output", type=str, help="Output file path (default: under Results/)")
    parser.add_argument("--task", type=str, default="MSMarco", help="Task name")
    parser.add_argument("--max_queries", type=int, default=None, help="Max number of queries to process")
    parser.add_argument("--top_k", type=int, default=None, help="Top K candidates per query to process")
    parser.add_argument("--candidates-num-each-anno", type=int, default=None,
                        dest="candidates_num_each_anno",
                        help="Max candidates per LLM annotation call; None = all at once (default)")
    parser.add_argument("--annotation-prompt-mode",
                        choices=["multi_doc_with_doc_id", "single_doc_no_doc_id"],
                        default="multi_doc_with_doc_id",
                        help="Annotation prompt contract. single_doc_no_doc_id implies hidden doc_id for one-candidate calls.")
    parser.add_argument("--hide-doc-id-for-single-doc", action="store_true",
                        help="Experimental: with one candidate per annotation call, hide doc_id from the LLM and backfill it in code.")
    args = parser.parse_args()
    if not args.output:
        args.output = f"Results/Annotated_{os.path.basename(args.input)}"
    llm_cfg = {'model': 'gpt-4o-mini', 'model_server': os.getenv('OPENAI_BASE_URL'), 'api_key': os.getenv('OPENAI_API_KEY')}
    tool = BatchCandidateAnnotator(task_name=args.task, llm_cfg=llm_cfg)
    params = json5.dumps({
        'input_path': args.input, 'output_path': args.output, 'task_name': args.task,
        'max_queries': args.max_queries, 'top_k': args.top_k,
        'candidates_num_each_anno': args.candidates_num_each_anno,
        'annotation_prompt_mode': args.annotation_prompt_mode,
        'hide_doc_id_for_single_doc': args.hide_doc_id_for_single_doc,
    }, ensure_ascii=False)
    tool.call(params, llm_cfg=llm_cfg)
