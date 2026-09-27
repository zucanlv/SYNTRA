import json
import os
import random

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from datasets import load_dataset


CACHE_DIR = "/data/share/project/shared_datasets"
DATASET_NAME = "mteb/nfcorpus"
SPLIT = "dev"
POSITIVE_SCORE = 2
SAMPLE_SIZE = 5
SEED = 42


def load_nfcorpus():
    """Load NFCorpus qrels, queries, and corpus from the shared HF cache."""
    common = {
        "path": DATASET_NAME,
        "cache_dir": CACHE_DIR,
        "download_mode": "reuse_dataset_if_exists",
    }
    qrels = load_dataset(name="default", **common)
    queries = load_dataset(name="queries", **common)
    corpus = load_dataset(name="corpus", **common)
    return qrels, queries, corpus


def select_positive_rows(rows, sample_size=SAMPLE_SIZE, seed=SEED):
    """Select deterministic score=2 pairs with distinct query IDs."""
    positive_rows = [row for row in rows if row["score"] == POSITIVE_SCORE]
    random.Random(seed).shuffle(positive_rows)

    selected = []
    seen_query_ids = set()
    for row in positive_rows:
        query_id = row["query-id"]
        if query_id in seen_query_ids:
            continue
        selected.append(row)
        seen_query_ids.add(query_id)
        if len(selected) == sample_size:
            break

    if len(selected) < sample_size:
        raise RuntimeError(
            f"{SPLIT} 中仅找到 {len(selected)} 个不同 query 的 "
            f"score={POSITIVE_SCORE} 正例，无法抽取 {sample_size} 条。"
        )
    return selected


def main():
    ds_default, ds_queries, ds_corpus = load_nfcorpus()

    query_by_id = {row["_id"]: row for row in ds_queries["queries"]}
    corpus_by_id = {row["_id"]: row for row in ds_corpus["corpus"]}
    selected_rows = select_positive_rows(ds_default[SPLIT])

    few_shot_examples = []

    print("=" * 80)
    print(
        f"NFCorpus default/{SPLIT}：固定种子 {SEED} 抽取 "
        f"{len(selected_rows)} 条不同 query 的 score={POSITIVE_SCORE} 正例"
    )
    print("=" * 80)

    for index, row in enumerate(selected_rows, start=1):
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]

        if query_id not in query_by_id:
            raise KeyError(f"queries 中不存在 query-id={query_id}")
        if corpus_id not in corpus_by_id:
            raise KeyError(f"corpus 中不存在 corpus-id={corpus_id}")

        query_row = query_by_id[query_id]
        corpus_row = corpus_by_id[corpus_id]
        few_shot_examples.append(
            {
                "query": query_row["text"],
                "Positive": corpus_row["text"],
            }
        )

        print(f"\n【第 {index} 条】")
        print(f"  query-id:  {query_id}")
        print(f"  corpus-id: {corpus_id}")
        print(f"  score:     {row['score']}")
        print("-" * 60)
        print("  QUERY 内容:")
        print(f"    {query_row['text']}")
        print("-" * 60)
        print("  CORPUS 内容:")
        print(f"    title: {corpus_row.get('title', '')}")
        print(f"    text:  {corpus_row['text']}")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("Few_Shot_Example 兼容格式（可复制到 Few_Shot_Example.py）：")
    print("=" * 80)
    print('    "nfcorpus": ', end="")
    print(json.dumps(few_shot_examples, indent=4, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
