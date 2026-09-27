import mteb

import json
import os
import sys

path = sys.argv[1]
results_list = os.listdir(path)
benchmark = "NanoBEIR"
if len(sys.argv) > 2:
    benchmark = sys.argv[2]
results = {}


def get_mteb_task_score(result):
    """Aggregate all split/subset scores with MTEB's default task-level mean."""
    scores = []
    for scores_list in result.get("scores", {}).values():
        for score_item in scores_list:
            main_score = score_item.get("main_score")
            if main_score is None:
                continue
            languages = score_item.get("languages", ["Unknown"])
            if not isinstance(languages, list):
                languages = [languages]
            if not languages:
                languages = ["Unknown"]
            scores.extend([main_score] * len(languages))
    if not scores:
        raise ValueError("No main_score found in result")
    return sum(scores) / len(scores)


def get_tasks(names: list[str] | None, languages: list[str] | None = None, benchmark: str | None = None):
    if benchmark:
        tasks = mteb.get_benchmark(benchmark).tasks
    else:
        tasks = mteb.get_tasks(languages=languages, tasks=names)

    return tasks

tasks = get_tasks(names=None, languages=None, benchmark=benchmark)
names = [t.metadata.name for t in tasks]
tasks = {name: task for name, task in zip(names, tasks)}

# print('names', names)
split_tasks = {}
task_files = {task.split(".json")[0]: task for task in results_list if task.endswith(".json")}
raw_results = {}
for name in names:
    if name not in task_files:
        continue
    meta = tasks[name].metadata 
    with open(os.path.join(path, task_files[name])) as f:
        result = json.load(f)
    # print('result', result)
    task_type = meta.type
    score = get_mteb_task_score(result)
    raw_results[name] = score
    results[name] = round(score * 100, 2)
    if task_type not in split_tasks:
        split_tasks[task_type] = []
    split_tasks[task_type].append(score)

final_scores = sum(raw_results.values()) / len(raw_results) * 100
missed_tasks = [name for name in names if name not in results]
print('missed tasks', missed_tasks)
print('final score', len(results), final_scores)
scores = []
for task_type in split_tasks:
    print(task_type, len(split_tasks[task_type]), sum(split_tasks[task_type]) / len(split_tasks[task_type]))
    score = sum(split_tasks[task_type]) / len(split_tasks[task_type])
    scores.append(score)
print('Mean(Type)', sum(scores) / len(scores))
for name in results:
    print(name, results[name])
