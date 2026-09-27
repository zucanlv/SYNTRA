"""
Retrieve top-K candidate passages for queries using FAISS index.

As a qwen_agent tool:
    from Query2Passge import Query2PassageTool
    tool = Query2PassageTool()
    result = tool.call(json5.dumps({
        'query_file': 'Results/DiverseQuery_xxx.json',
        'faiss_dir': '/path/to/faiss',
        'top_k': 20,
        'output_file': 'Results/Query2Passage_xxx.json',
        'faiss_gpu': True
    }))

As a standalone script:
    python Query2Passge.py \
      --query-file Results/DiverseQuery_Batch_xxx.json \
      --faiss-dir /data/share/project/zucan/data/faiss/arguana \
      --top-k 20 \
      --faiss-gpu --model-path /data/share/project/shared_models/bge-m3

FAISS directory layout (produced by Corpus_Build_Index.py):
    index.faiss   – FAISS IndexFlatIP
    id_map.pkl    – List[str], maps FAISS integer index → doc_id
    doc_dict.pkl  – Dict[str, {"title": str, "text": str}]
"""

import os
import json
import json5
import pickle
import torch
import numpy as np
import faiss
import logging
from typing import Any, Dict, List, Optional, Sequence
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime
from qwen_agent.tools.base import BaseTool, register_tool

os.makedirs("Logs", exist_ok=True)
os.makedirs("Results", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"Logs/Query2Passage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FAISS Resource Loader
# ---------------------------------------------------------------------------
def _normalize_faiss_gpu_ids(gpu_ids: Optional[Sequence[int]]) -> List[int]:
    if not gpu_ids:
        return [0]
    return [int(g) for g in gpu_ids]


def _valid_cuda_ids(gpu_ids: List[int]) -> List[int]:
    n = torch.cuda.device_count()
    return [g for g in gpu_ids if 0 <= g < n]


def _faiss_cpu_to_gpu_multi_sharded(cpu_index, gpu_ids: List[int]):
    """Shard *cpu_index* across GPUs using GpuResourcesVector + IntVector.

    ``index_cpu_to_gpu_multiple_py(ids, index, co)`` is **not** valid: the first
    argument must be GPU resource objects, not device ids. See faiss issue #878.
    Keeps ``referenced_objects`` on the returned index so resources are not GC'd.
    """
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    resources = [faiss.StandardGpuResources() for _ in gpu_ids]
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    for res, gid in zip(resources, gpu_ids):
        vres.push_back(res)
        vdev.push_back(int(gid))
    fn = getattr(faiss, "index_cpu_to_gpu_multiple", None)
    if fn is None:
        raise RuntimeError("faiss.index_cpu_to_gpu_multiple is not available (need faiss-gpu)")
    gpu_index = fn(vres, vdev, cpu_index, co)
    gpu_index.referenced_objects = resources
    return gpu_index


def load_faiss_resources(
    index_dir: str,
    use_gpu: bool = False,
    gpu_ids: Optional[Sequence[int]] = None,
):
    """Load FAISS index, id_map and doc_dict from *index_dir*.

    Returns ``(index, id_map, doc_dict)`` where:
    - ``id_map``   is ``List[str]``  (FAISS int index → doc_id)
    - ``doc_dict`` is ``Dict[str, {"title": str, "text": str}]``

    When *use_gpu* is True, the index is placed on GPU(s):
    - one ID → ``index_cpu_to_gpu``;
    - multiple IDs → ``index_cpu_to_gpu_multiple`` with *shard* (vectors split across GPUs).
    """
    index_path = os.path.join(index_dir, "index.faiss")
    id_map_path = os.path.join(index_dir, "id_map.pkl")
    doc_dict_path = os.path.join(index_dir, "doc_dict.pkl")

    if not os.path.exists(index_path):
        logger.error(f"FAISS index not found at {index_path}")
        raise FileNotFoundError(index_path)

    logger.info(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)

    gpu_ids = _normalize_faiss_gpu_ids(gpu_ids)
    if use_gpu and torch.cuda.is_available():
        valid = _valid_cuda_ids(gpu_ids)
        if len(valid) < len(gpu_ids):
            logger.warning(
                "Ignoring invalid GPU id(s); requested %s, using %s (cuda device count=%d)",
                gpu_ids,
                valid or [0],
                torch.cuda.device_count(),
            )
        gpu_ids = valid if valid else [0]

        faiss_ng = faiss.get_num_gpus()
        if faiss_ng == 0:
            logger.warning("FAISS reports 0 GPUs; staying on CPU for search.")
        else:
            gpu_ids = [g for g in gpu_ids if g < faiss_ng]
            if not gpu_ids:
                logger.warning("No FAISS-visible GPU ids; staying on CPU for search.")
            else:
                try:
                    if len(gpu_ids) == 1:
                        gid = gpu_ids[0]
                        logger.info("Moving FAISS index to GPU %d...", gid)
                        res = faiss.StandardGpuResources()
                        index = faiss.index_cpu_to_gpu(res, gid, index)
                        index.referenced_objects = [res]
                    else:
                        logger.info(
                            "Sharding FAISS index across GPUs %s (faiss_gpu)...",
                            gpu_ids,
                        )
                        index = _faiss_cpu_to_gpu_multi_sharded(index, gpu_ids)
                except Exception as e:
                    logger.warning(
                        "Failed to place FAISS index on GPU(s) %s: %s.",
                        gpu_ids,
                        e,
                    )
                    if len(gpu_ids) > 1:
                        try:
                            gid = gpu_ids[0]
                            logger.info(
                                "Retrying FAISS on single GPU %d (no sharding)...",
                                gid,
                            )
                            res = faiss.StandardGpuResources()
                            index = faiss.index_cpu_to_gpu(res, gid, index)
                            index.referenced_objects = [res]
                        except Exception as e2:
                            logger.warning(
                                "Single-GPU FAISS also failed: %s. Staying on CPU.",
                                e2,
                            )
                    else:
                        logger.warning("Staying on CPU for FAISS search.")

    logger.info(f"Loading ID map from {id_map_path}...")
    with open(id_map_path, "rb") as f:
        id_map = pickle.load(f)

    logger.info(f"Loading doc_dict from {doc_dict_path}...")
    with open(doc_dict_path, "rb") as f:
        doc_dict = pickle.load(f)

    logger.info(f"FAISS resources loaded: {index.ntotal} vectors, {len(doc_dict)} docs")
    return index, id_map, doc_dict


def _parse_gpu_ids_from_params(params_dict: Dict[str, Any]) -> List[int]:
    """Resolve GPU id list from tool params. ``gpu_ids`` overrides ``gpu_id`` when non-empty."""
    raw_ids = params_dict.get("gpu_ids")
    if raw_ids is not None:
        if isinstance(raw_ids, str):
            ids = [int(x.strip()) for x in raw_ids.split(",") if x.strip()]
        elif isinstance(raw_ids, (list, tuple)):
            ids = [int(x) for x in raw_ids]
        else:
            ids = []
        if ids:
            return ids
    gid = params_dict.get("gpu_id", None)
    if gid is None:
        return [0]
    if isinstance(gid, (list, tuple)):
        ids = [int(x) for x in gid]
        return ids if ids else [0]
    return [int(gid)]


# ---------------------------------------------------------------------------
# Query2Passage Tool
# ---------------------------------------------------------------------------
@register_tool('query2passage')
class Query2PassageTool(BaseTool):
    """Retrieve top-K candidate passages for each query using FAISS dense retrieval."""

    description = (
        "Given a JSON file of queries (produced by DiverseQueryGenerator), "
        "retrieve the top-K most similar passages from a FAISS index and "
        "attach candidate passage sets to each query. "
        "Returns the path to the output JSON file."
    )

    parameters = [
        {
            'name': 'query_file',
            'type': 'string',
            'description': 'Path to the JSON file containing queries (DiverseQuery output).',
            'required': True,
        },
        {
            'name': 'faiss_dir',
            'type': 'string',
            'description': 'Directory containing FAISS index (index.faiss, id_map.pkl, doc_dict.pkl).',
            'required': True,
        },
        {
            'name': 'top_k',
            'type': 'number',
            'description': 'Number of top candidates to retrieve per query. Default: 20.',
            'required': False,
        },
        {
            'name': 'output_file',
            'type': 'string',
            'description': 'Path to save the output JSON. If not provided, auto-generated from query_file name.',
            'required': False,
        },
        {
            'name': 'model_name',
            'type': 'string',
            'description': 'Embedding model name for encoding queries. Default: BAAI/bge-m3.',
            'required': False,
        },
        {
            'name': 'model_path',
            'type': 'string',
            'description': 'Local path to model directory (overrides model_name when set).',
            'required': False,
        },
        {
            'name': 'batch_size',
            'type': 'number',
            'description': 'Batch size for encoding queries. Default: 32.',
            'required': False,
        },
        {
            'name': 'device',
            'type': 'string',
            'description': 'Device to use for embedding model (cuda/cpu). Default: auto-detect.',
            'required': False,
        },
        {
            'name': 'faiss_gpu',
            'type': 'boolean',
            'description': 'Whether to use GPU for FAISS search if available. Default: False.',
            'required': False,
        },
        {
            'name': 'gpu_id',
            'type': 'number',
            'description': 'GPU device ID for embedding model and FAISS (e.g. 0, 1). None = use default (cuda:0).',
            'required': False,
        },
        {
            'name': 'gpu_ids',
            'type': 'array',
            'description': 'Optional list of GPU ids. When multiple ids are set, FAISS uses sharded multi-GPU; '
                           'the embedding model uses the first id. Overrides gpu_id when provided.',
            'required': False,
        },
        {
            'name': 'inject_origin_score',
            'type': 'boolean',
            'description': (
                'If true, pre-encode all unique origin docs and write each query\'s exact '
                'cosine similarity with its origin doc into q_obj["pos_score"]. '
                'Step 5 (HN Mine) then uses this value as raw_score for Branch A/B decisions, '
                'eliminating proxy-score bias when the origin doc falls outside top-k. Default: True.'
            ),
            'required': False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        params_dict = json5.loads(params)

        query_file = params_dict['query_file']
        faiss_dir = params_dict['faiss_dir']
        top_k = int(params_dict.get('top_k', 20))
        output_file = params_dict.get('output_file', None)
        model_name = params_dict.get('model_name', 'BAAI/bge-m3')
        model_path = params_dict.get('model_path', None)
        batch_size = int(params_dict.get('batch_size', 32))
        device = params_dict.get('device', None)
        faiss_gpu = bool(params_dict.get('faiss_gpu', False))
        gpu_ids_list = _parse_gpu_ids_from_params(params_dict)
        inject_origin_score = bool(params_dict.get('inject_origin_score', True))

        if not output_file:
            base_name = os.path.basename(query_file)
            name_without_ext = os.path.splitext(base_name)[0]
            output_file = f"Results/Query2Passage_{name_without_ext}.json"

        # Resolve device: explicit device string > first gpu in gpu_ids > default cuda/cpu
        embed_gpu = gpu_ids_list[0]
        if device:
            pass  # use as-is (cuda/cpu)
        elif torch.cuda.is_available():
            device = f"cuda:{embed_gpu}"
        else:
            device = "cpu"

        # 1. Load FAISS Resources
        logger.info("Loading FAISS resources...")
        index, id_map, doc_dict = load_faiss_resources(
            faiss_dir, use_gpu=faiss_gpu, gpu_ids=gpu_ids_list
        )

        # 2. Load Embedding Model
        model_source = model_path if model_path else model_name
        logger.info(f"Loading embedding model {model_source} on {device}...")
        model = SentenceTransformer(model_source, device=device)

        # 3. Load Queries
        logger.info(f"Loading queries from {query_file}...")
        with open(query_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        input_results = data.get("results", [])
        if not input_results:
            logger.warning("No results found in query file.")
            return json5.dumps({
                'output_file': output_file,
                'total_queries': 0,
                'status': 'no_queries_found'
            })

        # Flatten queries for batch processing, preserving references to original objects.
        # Each item carries source_doc_id and source_doc_text for origin-doc score injection.
        # source_doc_text is taken directly from item["text"] so that ID-format mismatches
        # between the input JSON (e.g. hash-style IDs) and the FAISS doc_dict (numeric IDs)
        # do not prevent pos_score computation.
        flattened_queries: List[Dict[str, Any]] = []
        for item in input_results:
            source_doc_id = item.get("doc_id", "")
            source_doc_text = item.get("doc", "")
            queries = item.get("queries", [])
            for idx, q_item in enumerate(queries):
                if isinstance(q_item, str):
                    q_item = {"query": q_item}
                    queries[idx] = q_item

                query_text = q_item.get("query", "")
                if query_text:
                    flattened_queries.append({
                        "query_obj": q_item,
                        "query_text": query_text,
                        "source_doc_id": source_doc_id,
                        "source_doc_text": source_doc_text,
                    })

        logger.info(f"Total queries to process: {len(flattened_queries)}")

        # 4a. Pre-encode origin docs (optional)
        # Collect unique source docs and encode their texts. Text is taken from item["text"]
        # rather than doc_dict so this works regardless of whether the item's doc_id matches
        # the FAISS index's key format (e.g. hash IDs vs numeric IDs in msmarco datasets).
        origin_embed_map: Dict[str, np.ndarray] = {}
        if inject_origin_score:
            seen: Dict[str, str] = {}  # doc_id → text, deduplicated
            for b in flattened_queries:
                sid, txt = b["source_doc_id"], b["source_doc_text"]
                if sid and txt and sid not in seen:
                    seen[sid] = txt
            if seen:
                unique_source_ids = list(seen.keys())
                origin_texts = [seen[sid] for sid in unique_source_ids]
                logger.info("Pre-encoding %d unique origin docs for pos_score injection ...",
                            len(unique_source_ids))
                origin_embs = model.encode(
                    origin_texts, batch_size=batch_size,
                    show_progress_bar=False, convert_to_numpy=True,
                )
                faiss.normalize_L2(origin_embs)
                origin_embed_map = dict(zip(unique_source_ids, origin_embs))
                logger.info("Origin doc pre-encoding done.")

        # 4b. Batch Retrieval
        for i in tqdm(range(0, len(flattened_queries), batch_size), desc="Retrieving"):
            batch = flattened_queries[i : i + batch_size]
            query_texts = [b["query_text"] for b in batch]

            embeddings = model.encode(
                query_texts, batch_size=len(batch),
                show_progress_bar=False, convert_to_numpy=True,
            )
            faiss.normalize_L2(embeddings)

            distances, indices = index.search(embeddings, top_k)

            for j, b in enumerate(batch):
                source_doc_id = b["source_doc_id"]
                query_obj = b["query_obj"]

                candidates = []

                for rank, idx_val in enumerate(indices[j]):
                    if idx_val == -1:
                        continue

                    doc_id = id_map[idx_val]
                    doc_entry = doc_dict.get(doc_id, {})
                    doc_content = doc_entry.get("doc", "")
                    score = float(distances[j][rank])

                    candidates.append({
                        "doc_id": doc_id,
                        "doc": doc_content,
                        "score": score,
                        "rank": rank + 1,
                        "is_original": (doc_id == source_doc_id),
                    })

                query_obj["candidates"] = candidates

                if inject_origin_score:
                    origin_emb = origin_embed_map.get(source_doc_id)
                    if origin_emb is not None:
                        query_obj["pos_score"] = float(np.dot(embeddings[j], origin_emb))

        # 5. Save Results
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        logger.info(f"Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Query2Passage done!")

        return json5.dumps({
            'output_file': output_file,
            'total_queries': len(flattened_queries),
            'status': 'success',
        })


# ---------------------------------------------------------------------------
# Standalone CLI entry-point
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve top-K candidate passages for queries.")
    parser.add_argument("--query-file", type=str, required=True, help="Path to the JSON file containing queries.")
    parser.add_argument("--faiss-dir", type=str, required=True,
                        help="Directory containing FAISS index and maps.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top candidates to retrieve (K).")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to save the output JSON. If None, generated from query-file name.")
    parser.add_argument("--model-name", type=str, default="BAAI/bge-m3", help="Embedding model name.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local path to model directory (overrides --model-name).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding queries.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu).")
    parser.add_argument("--faiss-gpu", action="store_true", help="Use GPU for FAISS search if possible.")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        dest="gpu_ids_str",
        help="Comma-separated GPU ids for FAISS (multi-GPU shards) and first id for embedding (e.g. 0,1,2).",
    )
    parser.add_argument("--gpu-id", type=int, default=None, dest="gpu_id",
                        help="Single GPU id (ignored if --gpu-ids is set). Default: 0.")
    parser.add_argument("--no-inject-origin-score", action="store_true", default=False,
                        dest="no_inject_origin_score",
                        help="Disable origin-doc cosine score injection (pos_score field). "
                             "Default: enabled (inject pos_score for each query).")
    args = parser.parse_args()

    tool = Query2PassageTool()
    params = {
        'query_file': args.query_file,
        'faiss_dir': args.faiss_dir,
        'top_k': args.top_k,
        'model_name': args.model_name,
        'batch_size': args.batch_size,
        'faiss_gpu': args.faiss_gpu,
        'inject_origin_score': not args.no_inject_origin_score,
    }
    if args.output_file:
        params['output_file'] = args.output_file
    if args.model_path:
        params['model_path'] = args.model_path
    if args.device:
        params['device'] = args.device
    if args.gpu_ids_str:
        params['gpu_ids'] = [int(x.strip()) for x in args.gpu_ids_str.split(",") if x.strip()]
    elif args.gpu_id is not None:
        params['gpu_id'] = args.gpu_id

    result = tool.call(json5.dumps(params))
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
