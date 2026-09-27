"""
Corpus index builder
====================
Encode a corpus with a sentence model and build a FAISS index.
Supports multiple corpus formats via :class:`corpus_reader.CorpusReader`.

Usage examples
--------------

# MSMARCO corpus (headerless TSV, auto-detected)
python Corpus_Build_Index.py \\
  --corpus /data/share/project/shared_datasets/MSMARCO/MSMARCO_Passage/collection.tsv \\
  --faiss-dir /data/share/project/zucan/data/faiss/ms_marco \\
  --model-path /data/share/project/shared_models/bge-m3 \\
  --batch-size 256

# MTEB dataset (HuggingFace Arrow, auto-detected)
python Corpus_Build_Index.py \\
  --corpus /data/share/project/shared_datasets/mteb/mteb___arguana/corpus \\
  --faiss-dir /data/share/project/zucan/data/faiss/arguana \\
  --batch-size 256

# Multi-GPU
python Corpus_Build_Index.py \\
  --corpus /path/to/corpus \\
  --faiss-dir /path/to/faiss \\
  --batch-size 256 --num-gpus 4

python /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/Corpus_Build_Index.py  --corpus /data/share/project/shared_datasets/mteb___trec-covid/corpus --faiss-dir /data/share/project/shared_datasets/DSA/faiss/trec-covid --batch-size 16 --num-gpus 6 --model-path /data/share/project/shared_models/bge-m3

python Corpus_Build_Index.py \\
  --corpus /path/to/corpus \\
  --faiss-dir /path/to/faiss \\
  --batch-size 256 --gpu-ids 0,1,2,3

# Explicit column mapping
python Corpus_Build_Index.py \\
  --corpus /path/to/corpus.jsonl \\
  --faiss-dir /path/to/faiss \\
  --id-column doc_id --text-column content --title-column ""

# Test run
python Corpus_Build_Index.py \
  --corpus /data/share/project/shared_datasets/mteb/mteb___arguana/corpus \
  --faiss-dir /data/share/project/zucan/data/faiss/test_20260303_arguana \
  --test --test-batch-limit 5 --model-path /data/share/project/shared_models/bge-m3
"""

import argparse
import json
import logging
import multiprocessing
import os
import pickle
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from corpus_reader import CorpusConfig, CorpusReader

# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class BuildIndexConfig:
    """All paths and hyper-parameters for corpus index building."""

    # Corpus
    corpus_path: str = ""
    corpus_format: str = "auto"
    id_column: Optional[str] = None
    text_column: Optional[str] = None
    title_column: Optional[str] = None  # None → auto;  "" → disable title

    # Output
    faiss_index_dir: str = ""
    log_dir: str = "Logs"

    # Model
    model_name: str = "BAAI/bge-m3"
    model_path: Optional[str] = None  # local path; overrides model_name when set
    device: Optional[str] = None

    # Batching & encoding
    batch_size: int = 256

    # Multi-GPU (data parallelism)
    num_gpus: int = 1
    gpu_ids: Optional[List[int]] = None

    # FAISS
    use_faiss_gpu: bool = True
    faiss_gpu_float16: bool = True
    # Large flat indices on GPU can OOM during index_gpu_to_cpu (extra temp alloc ~ index size).
    # Multi-GPU merge therefore keeps FAISS on CPU; encoding still uses GPUs in workers.
    faiss_merge_on_cpu: bool = True

    # Test / debug
    test_mode: bool = False
    test_batch_limit: int = 5

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- derived paths --

    @property
    def faiss_index_path(self) -> str:
        return os.path.join(self.faiss_index_dir, "index.faiss")

    @property
    def id_map_path(self) -> str:
        return os.path.join(self.faiss_index_dir, "id_map.pkl")

    @property
    def doc_dict_path(self) -> str:
        return os.path.join(self.faiss_index_dir, "doc_dict.pkl")

    @property
    def lines_consumed_path(self) -> str:
        return os.path.join(self.faiss_index_dir, "lines_consumed.pkl")

    # -- helpers --

    def get_corpus_config(self) -> CorpusConfig:
        return CorpusConfig(
            corpus_path=self.corpus_path,
            corpus_format=self.corpus_format,
            id_column=self.id_column,
            text_column=self.text_column,
            title_column=self.title_column,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "BuildIndexConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def get_encoding_gpu_ids(self) -> List[int]:
        if self.gpu_ids is not None and len(self.gpu_ids) > 0:
            ids = list(self.gpu_ids)
        else:
            ids = list(range(self.num_gpus))
        n = torch.cuda.device_count()
        if n == 0:
            return []
        valid = [i for i in ids if 0 <= i < n]
        if self.gpu_ids is not None and len(self.gpu_ids) > 0:
            return valid if valid else [0]
        return valid[: self.num_gpus] if valid else [0]


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args(args=None) -> BuildIndexConfig:
    """Parse CLI arguments (+ optional JSON config) into :class:`BuildIndexConfig`."""
    p = argparse.ArgumentParser(
        description="Build FAISS corpus index from a corpus using a sentence encoder."
    )
    p.add_argument("--config", type=str, default=None, help="JSON config file.")

    # Corpus
    p.add_argument(
        "--corpus", type=str, default=None,
        help="Path to corpus file or HuggingFace dataset directory.",
    )
    p.add_argument(
        "--corpus-format", type=str, default=None, dest="corpus_format",
        choices=["auto", "tsv", "csv", "jsonl", "parquet", "hf_dataset"],
        help="Corpus format (default: auto-detect).",
    )
    p.add_argument("--id-column", type=str, default=None, dest="id_column",
                   help="ID column name (default: auto-detect).")
    p.add_argument("--text-column", type=str, default=None, dest="text_column",
                   help="Text column name (default: auto-detect).")
    p.add_argument("--title-column", type=str, default=None, dest="title_column",
                   help='Title column (default: auto-detect; "" to disable).')

    # Output
    p.add_argument("--faiss-dir", type=str, default=None, dest="faiss_index_dir",
                   help="Directory for FAISS index, id_map, doc_dict.")
    p.add_argument("--log-dir", type=str, default=None)

    # Model
    p.add_argument("--model", type=str, default=None, dest="model_name",
                   help="Sentence-transformer model name (e.g. BAAI/bge-m3).")
    p.add_argument("--model-path", type=str, default=None, dest="model_path",
                   help="Local path to model directory (overrides --model when set).")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "auto"])

    # Batching
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-gpus", type=int, default=None, dest="num_gpus")
    p.add_argument("--gpu-ids", type=str, default=None, dest="gpu_ids_str",
                   help="Comma-separated GPU IDs (e.g. 0,1,2,3).")

    # FAISS
    p.add_argument("--no-faiss-gpu", action="store_true")
    p.add_argument("--faiss-fp32", action="store_true")
    p.add_argument(
        "--faiss-merge-gpu",
        action="store_true",
        help="Multi-GPU only: build merged FAISS index on GPU (faster add, risky for "
        "large corpora — can OOM when saving). Default: merge on CPU.",
    )

    # Test
    p.add_argument("--test", action="store_true", dest="test_mode")
    p.add_argument("--test-batch-limit", type=int, default=None)

    parsed = p.parse_args(args)

    # Start from config file if provided
    if parsed.config and os.path.isfile(parsed.config):
        with open(parsed.config, "r", encoding="utf-8") as f:
            config = BuildIndexConfig.from_dict(json.load(f))
    else:
        config = BuildIndexConfig()

    # Override with non-None CLI values
    if parsed.corpus is not None:
        config.corpus_path = parsed.corpus
    if parsed.corpus_format is not None:
        config.corpus_format = parsed.corpus_format
    if parsed.id_column is not None:
        config.id_column = parsed.id_column
    if parsed.text_column is not None:
        config.text_column = parsed.text_column
    if parsed.title_column is not None:
        config.title_column = parsed.title_column
    if parsed.faiss_index_dir is not None:
        config.faiss_index_dir = parsed.faiss_index_dir
    if parsed.log_dir is not None:
        config.log_dir = parsed.log_dir
    if parsed.model_name is not None:
        config.model_name = parsed.model_name
    if parsed.model_path is not None:
        config.model_path = parsed.model_path
    if parsed.device is not None:
        config.device = (
            "cuda"
            if parsed.device == "auto" and torch.cuda.is_available()
            else ("cpu" if parsed.device == "auto" else parsed.device)
        )
    if parsed.batch_size is not None:
        config.batch_size = parsed.batch_size
    if parsed.num_gpus is not None:
        config.num_gpus = parsed.num_gpus
    if getattr(parsed, "gpu_ids_str", None) is not None:
        config.gpu_ids = [
            int(x.strip()) for x in parsed.gpu_ids_str.split(",") if x.strip()
        ]
        config.num_gpus = len(config.gpu_ids)
    if parsed.no_faiss_gpu:
        config.use_faiss_gpu = False
    if parsed.faiss_fp32:
        config.faiss_gpu_float16 = False
    if getattr(parsed, "faiss_merge_gpu", False):
        config.faiss_merge_on_cpu = False
    if parsed.test_mode:
        config.test_mode = True
    if parsed.test_batch_limit is not None:
        config.test_batch_limit = parsed.test_batch_limit

    return config


# --------------------------------------------------------------------------- #
#  Logging                                                                     #
# --------------------------------------------------------------------------- #


def setup_logging(config: BuildIndexConfig):
    os.makedirs(config.log_dir, exist_ok=True)
    log_file = os.path.join(
        config.log_dir,
        f"Corpus_Build_Index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__), log_file


# --------------------------------------------------------------------------- #
#  FAISS helpers                                                               #
# --------------------------------------------------------------------------- #


def get_faiss_index_and_maps(config: BuildIndexConfig, logger: logging.Logger):
    """Load or create FAISS index, id_map, doc_dict, lines_consumed.

    Returns ``(index, id_map, doc_dict, lines_consumed)``.
    ``lines_consumed`` is *None* when ``lines_consumed.pkl`` is missing
    (backward compat – caller falls back to ``len(id_map)``).
    """
    os.makedirs(config.faiss_index_dir, exist_ok=True)
    id_map: List[str] = []
    doc_dict: dict = {}
    index = None
    lines_consumed = None

    if os.path.isfile(config.faiss_index_path):
        logger.info("Loading existing FAISS index from %s ...", config.faiss_index_path)
        index = faiss.read_index(config.faiss_index_path)
        with open(config.id_map_path, "rb") as f:
            id_map = pickle.load(f)
        with open(config.doc_dict_path, "rb") as f:
            doc_dict = pickle.load(f)
        if os.path.isfile(config.lines_consumed_path):
            with open(config.lines_consumed_path, "rb") as f:
                lines_consumed = pickle.load(f)
            logger.info(
                "Loaded %d vectors, doc_dict=%d entries, lines_consumed=%s",
                len(id_map), len(doc_dict), lines_consumed,
            )
        else:
            logger.info(
                "Loaded %d vectors, doc_dict=%d entries (no lines_consumed).",
                len(id_map), len(doc_dict),
            )
    return index, id_map, doc_dict, lines_consumed


def faiss_index_to_gpu(cpu_index, config: BuildIndexConfig):
    if not config.use_faiss_gpu or not torch.cuda.is_available():
        return cpu_index
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = config.faiss_gpu_float16
    return faiss.index_cpu_to_gpu(res, 0, cpu_index, co)


def faiss_index_to_cpu(gpu_index):
    if not isinstance(gpu_index, faiss.GpuIndex):
        return gpu_index
    return faiss.index_gpu_to_cpu(gpu_index)


# --------------------------------------------------------------------------- #
#  Multi-GPU worker                                                            #
# --------------------------------------------------------------------------- #


def _setup_worker_logger(log_file: str, gpu_id: int) -> logging.Logger:
    name = f"worker_gpu_{gpu_id}"
    lgr = logging.getLogger(name)
    lgr.setLevel(logging.INFO)
    if not lgr.handlers:
        handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s - %(levelname)s - [GPU {gpu_id}] %(message)s"
            )
        )
        lgr.addHandler(handler)
        lgr.propagate = False
    return lgr


def _encode_worker(args) -> str:
    """Encode a row-range of the corpus on a single GPU.

    Runs in a spawned process. Writes partial results (id_map, doc_dict,
    embeddings) to *temp_path* as a pickle.
    """
    (
        gpu_id,
        corpus_config_dict,
        start_row,
        end_row,
        skip_rows_base,
        model_name,
        model_path,
        batch_size,
        temp_path,
        log_file,
    ) = args

    wlog = _setup_worker_logger(log_file, gpu_id) if log_file else None

    def log_info(msg):
        if wlog:
            wlog.info(msg)

    model_source = model_path if model_path else model_name
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    log_info(f"Loading model {model_source} on {device} ...")
    model = SentenceTransformer(model_source, device=device)
    dim = model.get_sentence_embedding_dimension()

    id_map_part: List[str] = []
    doc_dict_part: dict = {}
    emb_list: list = []

    nrows = end_row - start_row
    if nrows <= 0:
        log_info(f"Empty row range [{start_row}, {end_row}), writing empty part.")
        with open(temp_path, "wb") as f:
            pickle.dump(
                {
                    "id_map": id_map_part,
                    "doc_dict": doc_dict_part,
                    "embeddings": np.zeros((0, dim), dtype=np.float32),
                },
                f,
            )
        return temp_path

    log_info(f"Encoding rows [{start_row}, {end_row}), total {nrows}")

    corpus_cfg = CorpusConfig.from_dict(corpus_config_dict)
    reader = CorpusReader(corpus_cfg)

    batch_count = 0
    log_interval = 50
    for batch, _ in reader.iter_batches(
        batch_size,
        skip_rows=skip_rows_base + start_row,
        max_rows=nrows,
    ):
        if not batch:
            continue

        doc_ids = [item["doc_id"] for item in batch]
        docs = [f"{item['title']} {item['text']}".strip() for item in batch]

        emb = model.encode(
            docs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        emb = np.asarray(emb, dtype=np.float32)
        faiss.normalize_L2(emb)

        id_map_part.extend(doc_ids)
        for did, doc in zip(doc_ids, docs):
            doc_dict_part[did] = {"doc": doc}
        emb_list.append(emb)

        batch_count += 1
        if batch_count % log_interval == 0:
            log_info(f"Progress: {len(id_map_part)} corpus entries ({batch_count} batches)")

    embeddings = (
        np.vstack(emb_list)
        if emb_list
        else np.zeros((0, dim), dtype=np.float32)
    )
    log_info(f"Done: {len(id_map_part)} corpus entries. Writing to {temp_path}.")
    with open(temp_path, "wb") as f:
        pickle.dump(
            {
                "id_map": id_map_part,
                "doc_dict": doc_dict_part,
                "embeddings": embeddings,
            },
            f,
        )
    return temp_path


# --------------------------------------------------------------------------- #
#  Main build                                                                  #
# --------------------------------------------------------------------------- #


def build_index(
    config: Optional[Union[BuildIndexConfig, dict]] = None,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Build or resume a FAISS corpus index.

    Returns the total number of corpus entries in the index after this run.

    When ``logger`` is provided (e.g. when called from main pipeline), that logger
    is used and the root logging config is left unchanged. When ``logger`` is None,
    logging is configured here via setup_logging().
    """
    if config is None:
        config = BuildIndexConfig()
    elif isinstance(config, dict):
        config = BuildIndexConfig.from_dict(config)

    if logger is None:
        logger, log_file = setup_logging(config)
    else:
        log_file = os.path.join(
            config.log_dir,
            f"Corpus_Build_Index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )

    try:
        os.makedirs(Path(config.faiss_index_dir).parent, exist_ok=True)

        index, id_map, doc_dict, lines_consumed = get_faiss_index_and_maps(
            config, logger
        )
        total_indexed = len(id_map)

        if lines_consumed is None:
            lines_consumed = total_indexed
            logger.warning(
                "No lines_consumed.pkl; using len(id_map) as resume position."
            )
        count_at_start = total_indexed

        # Create corpus reader
        reader = CorpusReader(config.get_corpus_config())
        logger.info("Corpus format: %s", reader.format)

        gpu_ids = config.get_encoding_gpu_ids()
        use_multi_gpu = len(gpu_ids) > 1 and torch.cuda.is_available()

        if use_multi_gpu:
            # ===== Multi-GPU path =====
            logger.info("Multi-GPU encoding with GPUs: %s", gpu_ids)
            logger.info("Counting corpus rows ...")
            total_lines = len(reader)
            logger.info("Total rows: %d", total_lines)

            rows_to_process = total_lines - lines_consumed
            if config.test_mode:
                rows_to_process = min(
                    rows_to_process, config.test_batch_limit * config.batch_size
                )
                logger.info("Test mode: limit to %d rows", rows_to_process)
            if rows_to_process <= 0:
                logger.info("No new rows to process; index unchanged.")
                return total_indexed

            N = len(gpu_ids)
            chunk_size = (rows_to_process + N - 1) // N
            ranges = []
            for i in range(N):
                s = i * chunk_size
                e = min(s + chunk_size, rows_to_process)
                if s >= e:
                    break
                ranges.append((s, e))

            for i, (s, e) in enumerate(ranges):
                logger.info(
                    "GPU %d (worker %d/%d): rows [%d, %d), count=%d  "
                    "(skip_rows_base=%d)",
                    gpu_ids[i], i + 1, len(ranges), s, e, e - s, lines_consumed,
                )

            tmpdir = tempfile.mkdtemp(prefix="faiss_build_", dir=config.faiss_index_dir)
            logger.info("Temp dir: %s", tmpdir)
            temp_paths = [
                os.path.join(tmpdir, f"part_{i}.pkl") for i in range(len(ranges))
            ]

            corpus_cfg_dict = config.get_corpus_config().to_dict()
            worker_args = [
                (
                    gpu_ids[i],
                    corpus_cfg_dict,
                    ranges[i][0],
                    ranges[i][1],
                    lines_consumed,
                    config.model_name,
                    config.model_path,
                    config.batch_size,
                    temp_paths[i],
                    log_file,
                )
                for i in range(len(ranges))
            ]

            start_time = time.time()
            logger.info("Starting %d encoding workers ...", len(ranges))
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=len(ranges)) as pool:
                pool.map(_encode_worker, worker_args)
            logger.info(
                "All workers finished in %.2fs. Merging ...", time.time() - start_time
            )

            # ---- merge partial results ----
            # Default: keep FAISS on CPU during merge so index_gpu_to_cpu does not need
            # an extra multi-GB GPU temp buffer (OOM on large flat indices). Use
            # --faiss-merge-gpu to merge on GPU (only for smaller corpora).
            use_gpu_for_merge = (
                config.use_faiss_gpu
                and torch.cuda.is_available()
                and not config.faiss_merge_on_cpu
            )
            if config.faiss_merge_on_cpu and config.use_faiss_gpu:
                logger.info(
                    "Multi-GPU merge: FAISS index on CPU (avoids large GPU temp alloc "
                    "when saving). Encoding used GPUs above."
                )

            dim = None
            if index is not None:
                if isinstance(index, faiss.GpuIndex):
                    index = faiss_index_to_cpu(index)
                dim = index.d
                if use_gpu_for_merge:
                    index = faiss_index_to_gpu(index, config)
                    logger.info("Moved existing FAISS index to GPU for merge.")

            num_parts = len(temp_paths)
            for i, path in enumerate(temp_paths):
                with open(path, "rb") as f:
                    part = pickle.load(f)
                emb = part["embeddings"]
                if emb.size == 0:
                    logger.info("Part %d/%d: empty, skip", i + 1, num_parts)
                    _safe_remove(path)
                    continue
                if dim is None:
                    dim = emb.shape[1]
                    index = faiss.IndexFlatIP(dim)
                    logger.info("Created new IndexFlatIP dim=%d", dim)
                    if use_gpu_for_merge:
                        index = faiss_index_to_gpu(index, config)
                index.add(emb)
                id_map.extend(part["id_map"])
                doc_dict.update(part["doc_dict"])
                logger.info(
                    "Part %d/%d: merged %d vectors, total so far: %d",
                    i + 1, num_parts, emb.shape[0], len(id_map),
                )
                _safe_remove(path)

            if index is not None and isinstance(index, faiss.GpuIndex):
                logger.info("Moving FAISS index back to CPU ...")
                index = faiss_index_to_cpu(index)
            _safe_remove(tmpdir, is_dir=True)

            total_indexed = len(id_map)
            elapsed = time.time() - start_time
            new_corpus_entries = total_indexed - count_at_start
            rate = new_corpus_entries / elapsed if elapsed > 0 else 0
            logger.info(
                "Multi-GPU: %d new corpus entries in %.2fs (%.2f docs/sec)",
                new_corpus_entries, elapsed, rate,
            )

            new_lines_consumed = lines_consumed + rows_to_process
            with open(config.lines_consumed_path, "wb") as f:
                pickle.dump(new_lines_consumed, f)

            if total_indexed > count_at_start:
                logger.info("Saving FAISS index and mappings ...")
                if index is not None:
                    faiss.write_index(index, config.faiss_index_path)
                with open(config.id_map_path, "wb") as f:
                    pickle.dump(id_map, f)
                with open(config.doc_dict_path, "wb") as f:
                    pickle.dump(doc_dict, f)

        else:
            # ===== Single-GPU path =====
            model_source = config.model_path if config.model_path else config.model_name
            logger.info("Loading model %s on %s ...", model_source, config.device)
            model = SentenceTransformer(model_source, device=config.device)
            dim = model.get_sentence_embedding_dimension()

            if index is None:
                logger.info("Creating new IndexFlatIP dim=%d", dim)
                index = faiss.IndexFlatIP(dim)
                index = faiss_index_to_gpu(index, config)
            else:
                index = faiss_index_to_gpu(index, config)

            logger.info(
                "Iterating corpus (skip_rows=%d) ...", lines_consumed,
            )
            start_time = time.time()
            lines_read_this_run = 0
            batch_count = 0
            consecutive_errors = 0
            _MAX_CONSECUTIVE_ERRORS = 3

            for batch, raw_count in reader.iter_batches(
                config.batch_size, skip_rows=lines_consumed
            ):
                if config.test_mode and batch_count >= config.test_batch_limit:
                    logger.info(
                        "Test mode limit reached (%d batches).", config.test_batch_limit
                    )
                    break

                batch_count += 1
                lines_read_this_run += raw_count

                if not batch:
                    continue

                doc_ids = [item["doc_id"] for item in batch]
                docs = [f"{item['title']} {item['text']}".strip() for item in batch]

                try:
                    embeddings = model.encode(
                        docs,
                        batch_size=config.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                    embeddings = np.asarray(embeddings, dtype=np.float32)
                    faiss.normalize_L2(embeddings)

                    index.add(embeddings)
                    id_map.extend(doc_ids)
                    for did, doc in zip(doc_ids, docs):
                        doc_dict[did] = {"doc": doc}

                    total_indexed += len(doc_ids)
                    consecutive_errors = 0
                    if batch_count % 20 == 0:
                        elapsed = time.time() - start_time
                        rate = (total_indexed - count_at_start) / elapsed if elapsed > 0 else 0
                        logger.info(
                            "Indexed %d corpus entries  (%.2f docs/sec)", total_indexed, rate
                        )
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(
                        "Error encoding batch (consecutive=%d/%d): %s",
                        consecutive_errors, _MAX_CONSECUTIVE_ERRORS, e,
                    )
                    lines_read_this_run -= raw_count
                    if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        logger.error(
                            "Reached %d consecutive encoding errors; aborting to prevent data loss.",
                            _MAX_CONSECUTIVE_ERRORS,
                        )
                        raise


            if total_indexed > count_at_start:
                logger.info("Saving FAISS index and mappings ...")
                index_to_save = faiss_index_to_cpu(index)
                faiss.write_index(index_to_save, config.faiss_index_path)
                with open(config.id_map_path, "wb") as f:
                    pickle.dump(id_map, f)
                with open(config.doc_dict_path, "wb") as f:
                    pickle.dump(doc_dict, f)
                new_lines_consumed = lines_consumed + lines_read_this_run
                with open(config.lines_consumed_path, "wb") as f:
                    pickle.dump(new_lines_consumed, f)
            else:
                logger.info("No new corpus entries; skipping save.")

    except Exception as e:
        logger.error("Global error: %s", e)
        raise

    logger.info("Index building completed!")
    logger.info("Final count: %d corpus entries in %s", total_indexed, config.faiss_index_dir)
    return total_indexed


# --------------------------------------------------------------------------- #
#  Utilities                                                                   #
# --------------------------------------------------------------------------- #


def _safe_remove(path: str, is_dir: bool = False):
    try:
        if is_dir:
            shutil.rmtree(path)
        else:
            os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
#  Entry                                                                       #
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    cfg = parse_args()
    build_index(cfg)
