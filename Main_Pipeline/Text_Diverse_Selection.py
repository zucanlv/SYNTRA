"""
Diverse text selection from a FAISS-indexed corpus
===================================================
Select semantically diverse texts by iteratively picking candidates and
removing their nearest neighbours from the available pool using Greedy Pagination.
Vectors are retrieved directly from the FAISS index to eliminate model encoding overhead.

Usage examples
--------------
# Quick test
python Text_Diverse_Selection.py \
  --faiss-dir /data/share/project/zucan/data/faiss/arguana \
  --test --test-limit 20

# Full run with target number
python Text_Diverse_Selection.py \
  --faiss-dir /data/share/project/zucan/data/faiss/arguana \
  --cos-thresh 0.6 --target-num 50000
"""

import argparse
import json
import logging
import math
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union

import faiss
import numpy as np
import torch
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#  Utilities                                                                   #
# --------------------------------------------------------------------------- #


class _NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class DiverseSelectionConfig:
    """All paths and hyper-parameters for diverse text selection."""

    # FAISS index directory (must contain index.faiss, id_map.pkl, doc_dict.pkl)
    faiss_index_dir: str = ""

    # Output
    output_dir: str = "Results"
    log_dir: str = "Logs"


    # Algorithm
    cos_thresh: float = 0.6
    top_k: int = 2000  
    target_num: Optional[int] = None

    # FAISS acceleration (IVF) — None = auto-select based on corpus size
    ivf_nlist: Optional[int] = None
    ivf_nprobe: Optional[int] = None
    ivf_train_sample: Optional[int] = None

    # Test / debug
    test_mode: bool = False
    test_select_limit: int = 20

    # Device for FAISS/torch (None = auto: cuda if available else cpu)
    device: Optional[str] = None
    # GPU device ID used for IVF index training (None = GPU 0 when device=cuda)
    gpu_id: Optional[int] = None

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

    # -- helpers --

    @classmethod
    def from_dict(cls, d: dict) -> "DiverseSelectionConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args(args=None) -> DiverseSelectionConfig:
    p = argparse.ArgumentParser(
        description="Select diverse texts from a FAISS-indexed corpus."
    )
    p.add_argument("--config", type=str, default=None, help="JSON config file.")

    # Paths
    p.add_argument(
        "--faiss-dir", type=str, default=None, dest="faiss_index_dir",
        help="Directory containing index.faiss, id_map.pkl, doc_dict.pkl.",
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=None)


    # Algorithm
    p.add_argument("--cos-thresh", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--target-num", type=int, default=None)

    # FAISS
    p.add_argument("--ivf-nlist", type=int, default=None)
    p.add_argument("--ivf-nprobe", type=int, default=None)
    p.add_argument("--ivf-train-sample", type=int, default=None)

    # Test
    p.add_argument("--test", action="store_true", dest="test_mode")
    p.add_argument("--test-limit", type=int, default=None, dest="test_select_limit")

    parsed = p.parse_args(args)

    if parsed.config and os.path.isfile(parsed.config):
        with open(parsed.config, "r", encoding="utf-8") as f:
            config = DiverseSelectionConfig.from_dict(json.load(f))
    else:
        config = DiverseSelectionConfig()

    for field in DiverseSelectionConfig.__dataclass_fields__:
        val = getattr(parsed, field, None)
        if val is not None:
            setattr(config, field, val)

    if getattr(parsed, "device", None) == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed.test_mode:
        config.test_mode = True

    return config


# --------------------------------------------------------------------------- #
#  Logging                                                                     #
# --------------------------------------------------------------------------- #


def _setup_logging(config: DiverseSelectionConfig):
    os.makedirs(config.log_dir, exist_ok=True)
    log_file = os.path.join(
        config.log_dir,
        f"Diverse_Selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  FAISS helpers                                                               #
# --------------------------------------------------------------------------- #


def _resolve_ivf_params(
    total_indexed: int, config: DiverseSelectionConfig, logger,
) -> tuple:
    """Determine nlist / nprobe / train_sample, auto-computing when None."""
    max_nlist = total_indexed // 39 + 1

    if config.ivf_nlist is not None:
        nlist = min(config.ivf_nlist, max_nlist)
        logger.info("ivf_nlist specified=%d, effective=%d (max feasible=%d)",
                     config.ivf_nlist, nlist, max_nlist)
    else:
        nlist = max(1, min(int(4 * math.sqrt(total_indexed)), max_nlist))
        logger.info("ivf_nlist auto-selected=%d  (4·√%d=%d, max feasible=%d)",
                     nlist, total_indexed,
                     int(4 * math.sqrt(total_indexed)), max_nlist)

    if config.ivf_nprobe is not None:
        nprobe = config.ivf_nprobe
    else:
        nprobe = max(8, int(math.sqrt(nlist)))
    logger.info("ivf_nprobe=%d  (%.2f%% of nlist)", nprobe, 100 * nprobe / nlist)

    min_train = 39 * nlist
    if config.ivf_train_sample is not None:
        train_sample = min(config.ivf_train_sample, total_indexed)
        if train_sample < min_train:
            logger.warning(
                "ivf_train_sample=%d < recommended minimum %d (39×nlist). "
                "Cluster quality may degrade.", train_sample, min_train)
    else:
        train_sample = min(256 * nlist, total_indexed)
    logger.info("ivf_train_sample=%d  (%.1f vectors/cluster)",
                train_sample, train_sample / max(nlist, 1))

    return nlist, nprobe, train_sample


def _prepare_accelerated_index(
    cpu_index, total_indexed: int, config: DiverseSelectionConfig, logger,
):
    """Convert IndexFlatIP → IndexIVFFlat for faster approximate search."""
    dim = cpu_index.d
    nlist, nprobe, train_sample = _resolve_ivf_params(
        total_indexed, config, logger,
    )

    quantizer = faiss.IndexFlatIP(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    sample_indices = np.random.choice(
        total_indexed, train_sample, replace=False,
    ).astype("int32")
    train_vecs = cpu_index.reconstruct_batch(sample_indices)

    use_gpu = config.device == "cuda" and faiss.get_num_gpus() > 0
    if use_gpu:
        gpu_id = config.gpu_id if config.gpu_id is not None else 0
        try:
            logger.info("Training IVF index (nlist=%d) on GPU %d ...", nlist, gpu_id)
            res = faiss.StandardGpuResources()
            res.setTempMemory(256 * 1024 * 1024)  # 256 MB — avoids the 1.5 GB default alloc
            gpu_ivf = faiss.index_cpu_to_gpu(res, gpu_id, index_ivf)
            gpu_ivf.train(train_vecs)
            index_ivf = faiss.index_gpu_to_cpu(gpu_ivf)
            del gpu_ivf, res
        except RuntimeError as exc:
            logger.warning(
                "GPU IVF training failed (%s); falling back to CPU training.", exc
            )
            use_gpu = False

    if not use_gpu:
        logger.info("Training IVF index (nlist=%d) on CPU ...", nlist)
        index_ivf.train(train_vecs)

    logger.info("Adding all vectors to IVF index ...")
    add_batch = 500_000
    for i in range(0, total_indexed, add_batch):
        end = min(i + add_batch, total_indexed)
        batch_vecs = cpu_index.reconstruct_batch(np.arange(i, end).astype("int32"))
        index_ivf.add(batch_vecs)
        logger.info("Added %d / %d vectors.", end, total_indexed)

    index_ivf.nprobe = nprobe
    logger.info("IVF index ready.  nprobe=%d", index_ivf.nprobe)

    # Enable direct map to allow O(1) reconstruction of original vectors by ID
    logger.info("Building direct map for fast vector reconstruction...")
    index_ivf.make_direct_map()

    return index_ivf


def _load_doc_dict(path: str, logger) -> dict:
    """Load doc_dict in ``{title, text}`` format."""
    with open(path, "rb") as f:
        doc_dict = pickle.load(f)
    logger.info("Loaded doc_dict: %d entries", len(doc_dict))
    return doc_dict


def _record_from_doc(doc_id: str, entry: dict) -> dict:
    """Single JSONL row schema shared by diverse selection and full export."""
    return {
        "doc_id": doc_id,
        "doc": entry["doc"],
    }


def export_all_indexed_texts(
    config: Optional[Union[DiverseSelectionConfig, dict]] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Write every document in ``id_map`` order to JSONL (same schema as ``select_diverse_texts``).

    Loads only ``id_map.pkl`` and ``doc_dict.pkl`` — does not read ``index.faiss`` or run FAISS.
    """
    if config is None:
        config = DiverseSelectionConfig()
    elif isinstance(config, dict):
        config = DiverseSelectionConfig.from_dict(config)

    if logger is None:
        logger = _setup_logging(config)

    try:
        os.makedirs(config.output_dir, exist_ok=True)

        if not os.path.isfile(config.id_map_path):
            raise FileNotFoundError(f"id_map not found: {config.id_map_path}")
        if not os.path.isfile(config.doc_dict_path):
            raise FileNotFoundError(f"doc_dict not found: {config.doc_dict_path}")

        logger.info(
            "export_all_indexed_texts: loading id_map + doc_dict from %s (no FAISS read)",
            config.faiss_index_dir,
        )
        with open(config.id_map_path, "rb") as f:
            id_map: List[str] = pickle.load(f)
        doc_dict = _load_doc_dict(config.doc_dict_path, logger)

        n_total = len(id_map)
        if not os.path.isfile(config.faiss_index_path):
            logger.info(
                "export_all_indexed_texts: %s missing; export does not need it. "
                "Step 4 Query2Passage still requires a built index under faiss_dir.",
                config.faiss_index_path,
            )

        limit = None
        if config.test_mode:
            limit = config.test_select_limit
            logger.info("export_all_indexed_texts: test_mode limit=%s", limit)

        output_file = os.path.join(
            config.output_dir,
            f"all_indexed_texts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )

        written = 0
        pbar_total = n_total if limit is None else min(n_total, limit)
        pbar = tqdm(total=pbar_total, desc="Export indexed texts")
        with open(output_file, "w", encoding="utf-8") as f_out:
            for doc_id in id_map:
                if limit is not None and written >= limit:
                    break
                entry = doc_dict[doc_id]
                record = _record_from_doc(doc_id, entry)
                f_out.write(json.dumps(record, ensure_ascii=False, cls=_NpEncoder) + "\n")
                written += 1
                pbar.update(1)
        pbar.close()

        logger.info("export_all_indexed_texts: wrote %d rows", written)
        logger.info("Output: %s", output_file)
        return output_file

    except Exception:
        logger.error("Error in export_all_indexed_texts", exc_info=True)
        raise


# --------------------------------------------------------------------------- #
#  Core algorithm                                                              #
# --------------------------------------------------------------------------- #


def select_diverse_texts(
    config: Optional[Union[DiverseSelectionConfig, dict]] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Select diverse texts from a FAISS-indexed corpus using Greedy Pagination.

    Eliminates model inference by directly reconstructing vectors from the index.
    Returns the path to the output JSONL file.

    When ``logger`` is provided (e.g. when called from main pipeline), that logger
    is used and the root logging config is left unchanged. When ``logger`` is None,
    logging is configured here via _setup_logging().
    """
    if config is None:
        config = DiverseSelectionConfig()
    elif isinstance(config, dict):
        config = DiverseSelectionConfig.from_dict(config)

    if logger is None:
        logger = _setup_logging(config)

    try:
        os.makedirs(config.output_dir, exist_ok=True)

        # ---- Load index artefacts ------------------------------------------------
        logger.info("Loading index artefacts from %s ...", config.faiss_index_dir)
        if not os.path.isfile(config.faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found: {config.faiss_index_path}")

        with open(config.id_map_path, "rb") as f:
            id_map: List[str] = pickle.load(f)
        doc_dict = _load_doc_dict(config.doc_dict_path, logger)

        total_indexed = len(id_map)
        docid_to_idx = {docid: i for i, docid in enumerate(id_map)}

        # ---- Build accelerated index --------------------------------------------
        logger.info("Reading raw Flat index from %s ...", config.faiss_index_path)
        raw_index = faiss.read_index(config.faiss_index_path)
        faiss_index = _prepare_accelerated_index(raw_index, total_indexed, config, logger)
        del raw_index


        # ---- Prepare candidate pool ----------------------------------------------
        available_ids_list = list(id_map)
        np.random.shuffle(available_ids_list)
        available_ids_set = set(available_ids_list)

        bitmap_size = (total_indexed + 7) // 8
        bitmap = np.full(bitmap_size, 255, dtype="uint8")
        selector = faiss.IDSelectorBitmap(total_indexed, faiss.swig_ptr(bitmap))
        search_params = faiss.SearchParametersIVF(sel=selector, nprobe=faiss_index.nprobe)

        logger.info("Corpus size: %d", total_indexed)
        logger.info("Parameters: cos_thresh=%.3f  target_num=%s",
                    config.cos_thresh, config.target_num)

        output_file = os.path.join(
            config.output_dir,
            f"diverse_texts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )

        success_count = 0
        pbar = tqdm(total=total_indexed, desc="Processed from pool")

        with open(output_file, "w", encoding="utf-8") as f_out:
            for current_idx in range(len(available_ids_list)):
                cid = available_ids_list[current_idx]
                if cid not in available_ids_set:
                    continue

                if config.test_mode and success_count >= config.test_select_limit:
                    logger.info("Test mode limit reached.")
                    break

                # --- A. Select the current candidate ---
                entry = doc_dict[cid]
                record = _record_from_doc(cid, entry)
                f_out.write(json.dumps(record, ensure_ascii=False, cls=_NpEncoder) + "\n")
                success_count += 1

                available_ids_set.discard(cid)
                c_idx = docid_to_idx[cid]
                bitmap[c_idx >> 3] &= 0xFF ^ (1 << (c_idx & 7))
                pbar.update(1)

                if config.target_num and success_count >= config.target_num:
                    break

                # --- B. Greedily prune high-similarity redundancies (single reconstruct + search) ---
                current_emb = faiss_index.reconstruct(int(c_idx)).reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(current_emb)

                while True:
                    scores, indices = faiss_index.search(
                        current_emb, config.top_k, params=search_params,
                    )
                    nb_indices = indices[0]
                    nb_scores = scores[0]
                    valid = nb_indices >= 0
                    nb_indices = nb_indices[valid]
                    nb_scores = nb_scores[valid]

                    if len(nb_scores) == 0:
                        break

                    if len(nb_scores) == config.top_k and nb_scores[-1] > config.cos_thresh:
                        to_remove_idxs = nb_indices
                        repeat_search = True
                    else:
                        mask = nb_scores > config.cos_thresh
                        to_remove_idxs = nb_indices[mask]
                        repeat_search = False

                    if len(to_remove_idxs) == 0:
                        break

                    removed_this_round = 0
                    for r_idx in to_remove_idxs:
                        r_docid = id_map[r_idx]
                        if r_docid in available_ids_set:
                            available_ids_set.discard(r_docid)
                            bitmap[r_idx >> 3] &= 0xFF ^ (1 << (r_idx & 7))
                            removed_this_round += 1
                    pbar.update(removed_this_round)

                    if repeat_search and removed_this_round == 0:
                        logger.warning("Pagination loop yielded no new removals. Breaking to prevent infinite loop.")
                        break
                    if not repeat_search:
                        break

                if success_count % 100 == 0:
                    logger.info(
                        "Selected %d diverse texts. Remaining pool: %d",
                        success_count, len(available_ids_set),
                    )

        pbar.close()

        # 4. Finalize and report
        if config.target_num is not None:
            if success_count >= config.target_num:
                logger.info("Successfully reached target_num: %d", config.target_num)
            else:
                logger.info(
                    "Corpus exhausted before reaching target_num! If you want more diverse docs, please increase the cos_thresh(But the 'diversity' may degrade)"
                    "Expected: %d, Actual collected: %d", 
                    config.target_num, success_count
                )
        
        logger.info("Done. Total diverse texts: %d", success_count)
        logger.info("Output: %s", output_file)
        return output_file

    except Exception:
        logger.error("Error in select_diverse_texts", exc_info=True)
        raise


# --------------------------------------------------------------------------- #
#  Entry                                                                       #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    cfg = parse_args()
    select_diverse_texts(cfg)