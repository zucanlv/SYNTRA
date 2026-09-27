"""
python /share/project/jianlv/Embedder-SOTA/evaluation/mmteb/code/summary.py \
/share/project/jianlv/Embedder-SOTA/evaluation/mmteb/results_output_folder/bge-multilingual-gemma2/no_model_name_available/no_revision_available
"""
import mteb
# from mteb.task_selection import results_to_dataframe

import json
import os
import sys

path = sys.argv[1]
results_list = os.listdir(path)
benchmark = "MTEB(eng, v2)"
if len(sys.argv) > 2:
    benchmark = sys.argv[2]

results = {}

def get_tasks(names: list[str] | None, languages: list[str] | None = None, benchmark: str | None = None):
    if benchmark:
        tasks = []
        for task in mteb.get_benchmark(benchmark).tasks:
            if task.metadata.type == "Retrieval":
                print("Including task:", task.metadata.name)
                tasks.append(task)
    else:
        tasks = mteb.get_tasks(languages=languages, tasks=names)
    return tasks


tasks = get_tasks(names=None, languages=None, benchmark=benchmark)
names = [t.metadata.name for t in tasks]
tasks = {name: task for name, task in zip(names, tasks)}

# print('names', names)
split_tasks = {}
for task in results_list:
    if task.split(".json")[0] not in names:
        continue
    name = task.split(".json")[0]
    meta = tasks[name].metadata 
    with open(os.path.join(path, task)) as f:
        result = json.load(f)
    # print('result', result)
    task_type = meta.type
    eval_split = list(result['scores'].keys())[0]
    
    score = sum([ele['main_score'] for ele in result['scores'][eval_split]]) / len(result['scores'][eval_split])
    results[name] = round(score * 100, 4)
    if task_type not in split_tasks:
        split_tasks[task_type] = []
    split_tasks[task_type].append(score)


final_scores = sum(results.values()) / len(results)
missed_tasks = [name for name in names if name not in results]
print("========================")
print('missed tasks', missed_tasks)
print('final score', len(results), final_scores)
scores = []
for task_type in split_tasks:
    print(task_type, len(split_tasks[task_type]), sum(split_tasks[task_type]) / len(split_tasks[task_type]))
    score = sum(split_tasks[task_type]) / len(split_tasks[task_type])
    scores.append(score)
print('Mean(Type)', sum(scores) / len(scores))
print("========================")
sorted_results = dict(sorted(results.items(), key=lambda item: item[0], reverse=False))
for name in sorted_results:
    print("------------------------")
    print(name, sorted_results[name])
