"""
python /share/project/jianlv/Embedder-SOTA/evaluation/mmteb/code/summary.py \
    /share/project/jianlv/Embedder-SOTA/evaluation/mmteb/results_output_folder/bge-multilingual-gemma2/no_model_name_available/no_revision_available \
    [benchmark] [model_name]
"""
import mteb
import json
import os
import sys

path = sys.argv[1]
results_list = os.listdir(path)
benchmark = "NanoBEIR"
if len(sys.argv) > 2:
    benchmark = sys.argv[2]

model_name = os.path.basename(os.path.normpath(path))
if len(sys.argv) > 3:
    model_name = sys.argv[3]

reranker = "NoReranker"
metrics = ["ndcg_at_10", "recall_at_100"]


def get_tasks(
    names: list[str] | None,
    languages: list[str] | None = None,
    benchmark: str | None = None,
):
    if benchmark:
        tasks = []
        for task in mteb.get_benchmark(benchmark).tasks:
            if task.metadata.type == "Retrieval":
                print("Including task:", task.metadata.name, file=sys.stderr)
                tasks.append(task)
    else:
        tasks = mteb.get_tasks(languages=languages, tasks=names)
    return tasks


tasks = get_tasks(names=None, languages=None, benchmark=benchmark)
task_lookup = {t.metadata.name: t for t in tasks}

# Parse results: per-task, per-metric averages
all_scores = {m: {} for m in metrics}
first_split = None

for fname in results_list:
    task_name = fname.split(".json")[0]
    if task_name not in task_lookup:
        continue

    with open(os.path.join(path, fname)) as f:
        result = json.load(f)

    eval_split = list(result["scores"].keys())[0]
    if first_split is None:
        first_split = eval_split
    scores_per_query = result["scores"][eval_split]

    for metric in metrics:
        vals = [q[metric] for q in scores_per_query]
        all_scores[metric][task_name] = sum(vals) / len(vals)

if not any(all_scores[m] for m in metrics):
    print("No results found!", file=sys.stderr)
    sys.exit(1)

# Determine task order (alphabetical by task name)
sorted_tasks = sorted(all_scores[metrics[0]].keys())

# Dataset column labels -> lowercase name + split
dataset_cols = []
for t in sorted_tasks:
    meta = task_lookup[t].metadata
    col = meta.name.lower().replace(" ", "-") + "-" + (first_split or "test")
    dataset_cols.append(col)

# Collect missed tasks for diagnostic output
missed_tasks = [name for name in task_lookup if name not in all_scores[metrics[0]]]
if missed_tasks:
    print("missed tasks:", missed_tasks, file=sys.stderr)

# Output transposed table: tasks as rows, metrics as columns
headers = ["Model", "Task"] + metrics
print("| " + " | ".join(headers) + " |")

aligns = [":----", ":----"] + [":---:"] * len(metrics)
print("| " + " | ".join(aligns) + " |")

for t in sorted_tasks:
    vals = [all_scores[m][t] for m in metrics]
    row = [model_name, dataset_cols[sorted_tasks.index(t)]] + [
        f"**{v * 100:.3f}**" for v in vals
    ]
    print("| " + " | ".join(row) + " |")

# Average row
avgs = [
    sum(all_scores[m].values()) / len(all_scores[m]) for m in metrics
]
row = [model_name, "average"] + [f"**{a * 100:.3f}**" for a in avgs]
print("| " + " | ".join(row) + " |")
print()
