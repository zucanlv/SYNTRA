"""Compare BEIR and NanoBEIR rankings across models.

Outputs a table: model name | BEIR rank | NanoBEIR rank | Rank diff | BEIR avg | NanoBEIR avg | diff
"""

from __future__ import annotations

import argparse
import logging
import warnings

import pandas as pd

import mteb
from mteb.load_results import load_results

logging.basicConfig(level=logging.WARNING)


def load_benchmark_scores(
    benchmark_name: str,
    results_repo: str,
) -> pd.Series:
    """Return a Series mapping model_name -> mean score for the given benchmark."""
    benchmark = mteb.get_benchmark(benchmark_name)
    if benchmark is None:
        raise ValueError(f"Benchmark '{benchmark_name}' not found.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        all_results = load_results(
            results_repo=results_repo,
            only_main_score=True,
            require_model_meta=False,
        )

    benchmark_results = (
        all_results.select_tasks(benchmark.tasks).join_revisions()
    )

    data = benchmark_results.to_dataframe(format="long")
    if data.empty:
        return pd.Series(dtype=float)

    per_task = data.pivot(index="model_name", columns="task_name", values="score")
    per_task = per_task.dropna(how="all")
    means = per_task.mean(axis=1, skipna=False)
    means.name = benchmark_name
    return means


def build_comparison_table(
    results_repo: str,
    min_tasks_fraction: float = 0.5,
) -> pd.DataFrame:
    """Build comparison table between BEIR and NanoBEIR."""
    print("Loading BEIR results...")
    beir_scores = load_benchmark_scores("BEIR", results_repo)

    print("Loading NanoBEIR results...")
    nano_scores = load_benchmark_scores("NanoBEIR", results_repo)

    # Combine into one dataframe; keep only models present in both benchmarks
    df = pd.DataFrame({"BEIR avg": beir_scores, "NanoBEIR avg": nano_scores})
    df = df.dropna()

    if df.empty:
        raise RuntimeError(
            "No models found with results in both BEIR and NanoBEIR. "
            "Check that the results repo contains data for both benchmarks."
        )

    # Multiply by 100 to get percentage scores
    df["BEIR avg"] = (df["BEIR avg"] * 100).round(2)
    df["NanoBEIR avg"] = (df["NanoBEIR avg"] * 100).round(2)
    df["diff"] = (df["NanoBEIR avg"] - df["BEIR avg"]).round(2)

    # Rank by average score (higher is better → ascending=False)
    df["BEIR rank"] = df["BEIR avg"].rank(method="min", ascending=False).astype(int)
    df["NanoBEIR rank"] = (
        df["NanoBEIR avg"].rank(method="min", ascending=False).astype(int)
    )
    df["Rank diff"] = df["NanoBEIR rank"] - df["BEIR rank"]

    # Clean up model name (strip HF org prefix for display)
    df.index.name = "model_name"
    df = df.reset_index()
    df["Model"] = df["model_name"].apply(lambda n: n.split("/")[-1])

    df = df.sort_values("BEIR rank")

    result = df[
        ["Model", "BEIR rank", "NanoBEIR rank", "Rank diff", "BEIR avg", "NanoBEIR avg", "diff"]
    ].reset_index(drop=True)

    return result


def print_markdown_table(df: pd.DataFrame) -> None:
    """Print DataFrame as a markdown table."""
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    print(header)
    print(separator)
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if col == "Rank diff":
                cells.append(f"{val:+d}")
            elif col == "diff":
                cells.append(f"{val:+.2f}")
            else:
                cells.append(str(val))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare BEIR and NanoBEIR rankings across models."
    )
    parser.add_argument(
        "--results-repo",
        default="https://gh-proxy.org/https://github.com/embeddings-benchmark/results.git",
        help="Path or URL to the results repository.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the table as CSV.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "plain"],
        default="markdown",
        help="Output format (default: markdown).",
    )
    args = parser.parse_args()

    table = build_comparison_table(results_repo=args.results_repo)

    if args.format == "markdown":
        print_markdown_table(table)
    elif args.format == "csv":
        print(table.to_csv(index=False))
    else:
        print(table.to_string(index=False))

    if args.output:
        table.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
