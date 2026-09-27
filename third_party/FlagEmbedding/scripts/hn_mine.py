"""
Hard negative mining (hn_mine).

Two-phase (recommended when FAISS GPU OOMs after encoding):
  1) encode_only: compute embeddings, save to --hn_mine_cache_dir, exit (PyTorch process ends).
  2) search_only: load cache, build FAISS + write output (no FlagEmbedding / no model load).

Single-process: --hn_mine_mode full (default).
"""
import gc
import json
import os
import random
import numpy as np
from tqdm import tqdm
from typing import Any, List, Optional, Tuple
from dataclasses import dataclass, field

import faiss
from transformers import HfArgumentParser

# torch, FlagAutoModel imported lazily in encode paths so `search_only` avoids loading PyTorch.


CACHE_MANIFEST = "manifest.json"
CACHE_PASSAGE_EMB = "passage_emb.npy"
CACHE_QUERY_EMB = "query_emb.npy"
CACHE_CORPUS_JSONL = "corpus.jsonl"
CACHE_TRAIN_JSONL = "train.jsonl"


@dataclass
class DataArgs:
    """
    Data arguments for hard negative mining.
    """
    input_file: str = field(
        default=None, metadata={"help": "The input file for hard negative mining (jsonl). Not used when hn_mine_mode=search_only."}
    )
    output_file: str = field(
        default=None, metadata={"help": "The output file for hard negative mining. Ignored for encode_only if manifest stores it; can override in search_only."}
    )
    candidate_pool: Optional[str] = field(
        default=None, metadata={"help": "The candidate pool for hard negative mining. If provided, it should be a jsonl file, each line is a dict with a key 'text'."}
    )
    range_for_sampling: str = field(
        default="10-210", metadata={"help": "The range to sample negatives."}
    )
    negative_number: int = field(
        default=15, metadata={"help": "The number of negatives."}
    )
    use_gpu_for_searching: bool = field(
        default=False, metadata={"help": "Whether to use faiss-gpu for searching."}
    )
    search_batch_size: int = field(
        default=64, metadata={"help": "The batch size for searching."}
    )
    hn_mine_mode: str = field(
        default="full",
        metadata={"help": "full: encode + FAISS in one process. encode_only: save embeddings to hn_mine_cache_dir and exit. search_only: load cache from hn_mine_cache_dir (second process; no embedder).", "choices": ["full", "encode_only", "search_only"]}
    )
    hn_mine_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for two-phase runs: encode_only writes passage/query npy + jsonl here; search_only reads from here."}
    )


@dataclass
class ModelArgs:
    """
    Model arguments for embedder.
    """
    embedder_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The embedder name or path. Not used when hn_mine_mode=search_only."}
    )
    embedder_model_class: Optional[str] = field(
        default=None, metadata={"help": "The embedder model class. Available classes: ['encoder-only-base', 'encoder-only-m3', 'decoder-only-base', 'decoder-only-icl']. Default: None. For the custom model, you need to specifiy the model class.", "choices": ["encoder-only-base", "encoder-only-m3", "decoder-only-base", "decoder-only-icl"]}
    )
    normalize_embeddings: bool = field(
        default=True, metadata={"help": "whether to normalize the embeddings"}
    )
    pooling_method: str = field(
        default="cls", metadata={"help": "The pooling method fot the embedder."}
    )
    use_fp16: bool = field(
        default=True, metadata={"help": "whether to use fp16 for inference"}
    )
    devices: Optional[str] = field(
        default=None, metadata={"help": "Devices to use for inference.", "nargs": "+"}
    )
    query_instruction_for_retrieval: Optional[str] = field(
        default=None, metadata={"help": "Instruction for query"}
    )
    query_instruction_format_for_retrieval: str = field(
        default="{}{}", metadata={"help": "Format for query instruction"}
    )
    examples_for_task: Optional[str] = field(
        default=None, metadata={"help": "Examples for task"}
    )
    examples_instruction_format: str = field(
        default="{}{}", metadata={"help": "Format for examples instruction"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Cache directory for models."}
    )
    # ================ for inference ===============
    batch_size: int = field(
        default=3000, metadata={"help": "Batch size for inference."}
    )
    embedder_query_max_length: int = field(
        default=512, metadata={"help": "Max length for query."}
    )
    embedder_passage_max_length: int = field(
        default=512, metadata={"help": "Max length for passage."}
    )
    
    def __post_init__(self):
        # replace "\\n" with "\n"
        if "\\n" in self.query_instruction_format_for_retrieval:
            self.query_instruction_format_for_retrieval = self.query_instruction_format_for_retrieval.replace("\\n", "\n")
        if "\\n" in self.examples_instruction_format:
            self.examples_instruction_format = self.examples_instruction_format.replace("\\n", "\n")


def release_gpu_memory_after_embedder() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)
        torch.cuda.empty_cache()


def create_index(embeddings: np.ndarray, use_gpu: bool = False):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def batch_search(
    index: faiss.Index,
    query: np.ndarray,
    topk: int = 200,
    batch_size: int = 64
):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs


def get_corpus(candidate_pool: str):
    corpus = []
    with open(candidate_pool, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            corpus.append(line['text'])
    return corpus


def load_input_and_corpus(
    input_file: str,
    candidate_pool: Optional[str],
) -> Tuple[List[str], List[str], List[dict]]:
    corpus_acc: List[str] = []
    queries: List[str] = []
    train_data: List[dict] = []
    for line in open(input_file, encoding="utf-8"):
        line = json.loads(line.strip())
        train_data.append(line)
        corpus_acc.extend(line['pos'])
        if 'neg' in line:
            corpus_acc.extend(line['neg'])
        queries.append(line['query'])

    if candidate_pool is not None:
        if not isinstance(candidate_pool, list):
            candidate_pool = get_corpus(candidate_pool)
        corpus = list(set(candidate_pool))
    else:
        corpus = list(set(corpus_acc))

    return corpus, queries, train_data


def save_encode_cache(
    cache_dir: str,
    p_vecs: np.ndarray,
    q_vecs: np.ndarray,
    corpus: List[str],
    train_data: List[dict],
    data_args: DataArgs,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, CACHE_PASSAGE_EMB), np.asarray(p_vecs, dtype=np.float32))
    np.save(os.path.join(cache_dir, CACHE_QUERY_EMB), np.asarray(q_vecs, dtype=np.float32))
    corpus_path = os.path.join(cache_dir, CACHE_CORPUS_JSONL)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for t in corpus:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    train_path = os.path.join(cache_dir, CACHE_TRAIN_JSONL)
    with open(train_path, "w", encoding="utf-8") as f:
        for row in train_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest: dict[str, Any] = {
        "version": 1,
        "sample_range": [int(x) for x in data_args.range_for_sampling.split("-")],
        "negative_number": data_args.negative_number,
        "output_file": data_args.output_file,
        "use_gpu_for_searching": data_args.use_gpu_for_searching,
        "search_batch_size": data_args.search_batch_size,
        "range_for_sampling": data_args.range_for_sampling,
    }
    with open(os.path.join(cache_dir, CACHE_MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved encode cache to {cache_dir} ({len(corpus)} passages, {len(train_data)} train lines).")


def load_encode_cache(cache_dir: str, output_override: Optional[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[dict], List[int], int, str, bool, int]:
    manifest_path = os.path.join(cache_dir, CACHE_MANIFEST)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    p_vecs = np.load(os.path.join(cache_dir, CACHE_PASSAGE_EMB))
    q_vecs = np.load(os.path.join(cache_dir, CACHE_QUERY_EMB))
    corpus: List[str] = []
    with open(os.path.join(cache_dir, CACHE_CORPUS_JSONL), "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line.strip())["text"])
    train_data: List[dict] = []
    with open(os.path.join(cache_dir, CACHE_TRAIN_JSONL), "r", encoding="utf-8") as f:
        for line in f:
            train_data.append(json.loads(line.strip()))
    sample_range = manifest["sample_range"]
    negative_number = manifest["negative_number"]
    output_file = output_override if output_override else manifest["output_file"]
    use_gpu = manifest["use_gpu_for_searching"]
    search_batch_size = manifest["search_batch_size"]
    return p_vecs, q_vecs, corpus, train_data, sample_range, negative_number, output_file, use_gpu, search_batch_size


def run_faiss_and_write(
    p_vecs: np.ndarray,
    q_vecs: np.ndarray,
    corpus: List[str],
    train_data: List[dict],
    sample_range: List[int],
    negative_number: int,
    output_file: str,
    use_gpu: bool,
    search_batch_size: int,
) -> None:
    print("create index and search------------------")
    index = create_index(p_vecs, use_gpu=use_gpu)
    _, all_inxs = batch_search(index, q_vecs, topk=sample_range[-1], batch_size=search_batch_size)
    assert len(all_inxs) == len(train_data)

    for i, data in enumerate(train_data):
        query = data["query"]
        inxs = all_inxs[i][sample_range[0]: sample_range[1]]
        filtered_inx = []
        for inx in inxs:
            if inx == -1:
                break
            if corpus[inx] not in data["pos"] and corpus[inx] != query:
                filtered_inx.append(inx)

        if len(filtered_inx) > negative_number:
            filtered_inx = random.sample(filtered_inx, negative_number)
        data["neg"] = [corpus[inx] for inx in filtered_inx]

    with open(output_file, "w", encoding="utf-8") as f:
        for data in train_data:
            if len(data["neg"]) < negative_number:
                samples = random.sample(corpus, negative_number - len(data["neg"]) + len(data["pos"]))
                samples = [sent for sent in samples if sent not in data["pos"]]
                data["neg"].extend(samples[: negative_number - len(data["neg"])])
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def find_knn_neg(
    model: Any,
    input_file: str,
    output_file: str,
    candidate_pool: Optional[str] = None,
    sample_range: str = "10-210",
    negative_number: int = 15,
    use_gpu: bool = False,
    search_batch_size: int = 64,
):
    corpus, queries, train_data = load_input_and_corpus(input_file, candidate_pool)

    print(f"inferencing embedding for corpus (number={len(corpus)})--------------")
    p_vecs = model.encode(corpus)
    print(f"inferencing embedding for queries (number={len(queries)})--------------")
    q_vecs = model.encode_queries(queries)

    if isinstance(p_vecs, dict):
        p_vecs = p_vecs["dense_vecs"]
    if isinstance(q_vecs, dict):
        q_vecs = q_vecs["dense_vecs"]

    del model
    release_gpu_memory_after_embedder()

    sr = [int(x) for x in sample_range.split("-")]
    run_faiss_and_write(
        p_vecs=np.asarray(p_vecs, dtype=np.float32),
        q_vecs=np.asarray(q_vecs, dtype=np.float32),
        corpus=corpus,
        train_data=train_data,
        sample_range=sr,
        negative_number=negative_number,
        output_file=output_file,
        use_gpu=use_gpu,
        search_batch_size=search_batch_size,
    )


def load_model(model_args: ModelArgs):
    from FlagEmbedding import FlagAutoModel
    model = FlagAutoModel.from_finetuned(
        model_name_or_path=model_args.embedder_name_or_path,
        model_class=model_args.embedder_model_class,
        normalize_embeddings=model_args.normalize_embeddings,
        pooling_method=model_args.pooling_method,
        use_fp16=model_args.use_fp16,
        query_instruction_for_retrieval=model_args.query_instruction_for_retrieval,
        query_instruction_format=model_args.query_instruction_format_for_retrieval,
        devices=model_args.devices,
        examples_for_task=model_args.examples_for_task,
        examples_instruction_format=model_args.examples_instruction_format,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
        batch_size=model_args.batch_size,
        query_max_length=model_args.embedder_query_max_length,
        passage_max_length=model_args.embedder_passage_max_length,
    )
    return model


def main(data_args: DataArgs, model_args: ModelArgs):
    mode = data_args.hn_mine_mode
    cache_dir = data_args.hn_mine_cache_dir

    if mode == "search_only":
        if not cache_dir:
            raise ValueError("hn_mine_mode=search_only requires --hn_mine_cache_dir.")
        p_vecs, q_vecs, corpus, train_data, sample_range, neg_n, out_file, use_gpu, sb = load_encode_cache(
            cache_dir, data_args.output_file
        )
        run_faiss_and_write(
            p_vecs=p_vecs,
            q_vecs=q_vecs,
            corpus=corpus,
            train_data=train_data,
            sample_range=sample_range,
            negative_number=neg_n,
            output_file=out_file,
            use_gpu=use_gpu,
            search_batch_size=sb,
        )
        return

    if not data_args.input_file:
        raise ValueError("hn_mine_mode full/encode_only requires --input_file.")
    if not model_args.embedder_name_or_path:
        raise ValueError("hn_mine_mode full/encode_only requires --embedder_name_or_path.")

    model = load_model(model_args)

    if mode == "encode_only":
        if not cache_dir:
            raise ValueError("hn_mine_mode=encode_only requires --hn_mine_cache_dir.")
        if not data_args.output_file:
            raise ValueError("hn_mine_mode=encode_only requires --output_file (stored in manifest for search_only).")
        corpus, queries, train_data = load_input_and_corpus(data_args.input_file, data_args.candidate_pool)
        print(f"inferencing embedding for corpus (number={len(corpus)})--------------")
        p_vecs = model.encode(corpus)
        print(f"inferencing embedding for queries (number={len(queries)})--------------")
        q_vecs = model.encode_queries(queries)
        if isinstance(p_vecs, dict):
            p_vecs = p_vecs["dense_vecs"]
        if isinstance(q_vecs, dict):
            q_vecs = q_vecs["dense_vecs"]
        save_encode_cache(
            cache_dir,
            np.asarray(p_vecs, dtype=np.float32),
            np.asarray(q_vecs, dtype=np.float32),
            corpus,
            train_data,
            data_args,
        )
        return

    # full
    find_knn_neg(
        model=model,
        input_file=data_args.input_file,
        output_file=data_args.output_file,
        candidate_pool=data_args.candidate_pool,
        sample_range=data_args.range_for_sampling,
        negative_number=data_args.negative_number,
        use_gpu=data_args.use_gpu_for_searching,
        search_batch_size=data_args.search_batch_size,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((
        DataArgs,
        ModelArgs
    ))
    data_args, model_args = parser.parse_args_into_dataclasses()
    data_args: DataArgs
    model_args: ModelArgs
    if data_args.hn_mine_mode in ("full", "encode_only"):
        if not data_args.input_file:
            raise ValueError("--input_file is required for hn_mine_mode=full or encode_only.")
        if not data_args.output_file:
            raise ValueError("--output_file is required for hn_mine_mode=full or encode_only.")
    main(data_args, model_args)
