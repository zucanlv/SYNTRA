#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""查看 CoIR-Retrieval/cosqa 的 5 个 golden query 和 golden positive。"""

from datasets import load_from_disk


DATASET_DIR = "/data/share/project/shared_datasets/DSA/datasets/CoIR-Retrieval__cosqa"
LIMIT = 5


def build_index(dataset):
    return {row["_id"]: row for row in dataset}


def first_split(dataset_dict, preferred=("test", "valid", "train")):
    for split in preferred:
        if split in dataset_dict:
            return split, dataset_dict[split]
    split = next(iter(dataset_dict))
    return split, dataset_dict[split]


def print_golden_examples(limit=LIMIT):
    qrels = load_from_disk(f"{DATASET_DIR}/default")
    queries = load_from_disk(f"{DATASET_DIR}/queries")["queries"]
    corpus = load_from_disk(f"{DATASET_DIR}/corpus")["corpus"]

    query_by_id = build_index(queries)
    corpus_by_id = build_index(corpus)
    split_name, qrel_split = first_split(qrels)

    print(f"数据集: CoIR-Retrieval/cosqa")
    print(f"qrels split: {split_name}")
    print("=" * 100)

    shown = 0
    for qrel in qrel_split:
        if qrel["score"] <= 0:
            continue

        query = query_by_id.get(qrel["query-id"])
        positive = corpus_by_id.get(qrel["corpus-id"])
        if query is None or positive is None:
            continue

        shown += 1
        print(f"第{shown}条")
        print(f"query-id: {qrel['query-id']}")
        print(f"golden query:\n{query['text']}")
        print(f"corpus-id: {qrel['corpus-id']}")
        print(f"golden positive:\n{positive['text']}")
        print("-" * 100)

        if shown >= limit:
            break

    if shown == 0:
        print("没有找到 score > 0 且 query/corpus 都能匹配到的样例。")


if __name__ == "__main__":
    print_golden_examples()

'''
```python
    "coir-cosqa": [
        {
                "query": "sort by a token in string python",
                "Positive": "def _process_and_sort(s, force_ascii, full_process=True):\n    \"\"\"Return a cleaned string with token sorted.\"\"\"\n    # pull tokens\n    ts = utils.full_process(s, force_ascii=force_ascii) if full_process else s\n    tokens = ts.split()\n\n    # sort tokens and join\n    sorted_string = u\" \".join(sorted(tokens))\n    return sorted_string.strip()"
        },
        {
                "query": "python check file is readonly",
                "Positive": "def is_readable(filename):\n    \"\"\"Check if file is a regular file and is readable.\"\"\"\n    return os.path.isfile(filename) and os.access(filename, os.R_OK)"
        },
        {
                "query": "declaring empty numpy array in python",
                "Positive": "def empty(self, name, **kwargs):\n        \"\"\"Create an array. Keyword arguments as per\n        :func:`zarr.creation.empty`.\"\"\"\n        return self._write_op(self._empty_nosync, name, **kwargs)"
        },
        {
                "query": "test for iterable is string in python",
                "Positive": "def is_iterable_but_not_string(obj):\n    \"\"\"\n    Determine whether or not obj is iterable but not a string (eg, a list, set, tuple etc).\n    \"\"\"\n    return hasattr(obj, '__iter__') and not isinstance(obj, str) and not isinstance(obj, bytes)"
        },
        {
                "query": "python print results of query loop",
                "Positive": "def print_runs(query):\n    \"\"\" Print all rows in this result query. \"\"\"\n\n    if query is None:\n        return\n\n    for tup in query:\n        print((\"{0} @ {1} - {2} id: {3} group: {4}\".format(\n            tup.end, tup.experiment_name, tup.project_name,\n            tup.experiment_group, tup.run_group)))"
        }
    ],
```
'''
