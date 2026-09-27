"""Summarize MTEB task JSON files without consulting the benchmark registry."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def task_score(result: dict) -> float | None:
    scores = []
    for split_scores in result.get("scores", {}).values():
        for score_item in split_scores:
            main_score = score_item.get("main_score")
            if main_score is None:
                continue
            languages = score_item.get("languages", ["Unknown"])
            if not isinstance(languages, list):
                languages = [languages]
            scores.extend([float(main_score)] * max(len(languages), 1))
    if not scores:
        return None
    return sum(scores) / len(scores)


def main() -> None:
    args = parse_args()
    results = {}
    for result_path in sorted(args.result_dir.glob("*.json")):
        with result_path.open("r", encoding="utf-8") as input_file:
            result = json.load(input_file)
        score = task_score(result)
        if score is not None:
            results[result.get("task_name", result_path.stem)] = score

    if not results:
        raise ValueError(f"No task main_score values found in {args.result_dir}")

    mean_score = sum(results.values()) / len(results) * 100
    print(f"final score {len(results)} {mean_score}")
    for task_name, score in results.items():
        print(f"{task_name} {round(score * 100, 2)}")


if __name__ == "__main__":
    main()
