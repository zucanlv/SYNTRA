#!/usr/bin/env python3
"""Download selected CoIR-Retrieval datasets and save them locally."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATASETS = (
    "CoIR-Retrieval/cosqa",
    "CoIR-Retrieval/synthetic-text2sql",
    "CoIR-Retrieval/codetrans-dl",
)
DEFAULT_CONFIGS = ("default", "corpus", "queries")
DEFAULT_OUTPUT_DIR = Path("/data/share/project/shared_datasets/DSA/datasets")


def repo_to_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CoIR-Retrieval datasets from Hugging Face."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Base directory to write datasets into. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset repo IDs to download.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help=f"Configs to download for each dataset. Default: {' '.join(DEFAULT_CONFIGS)}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Optional number of processes for save_to_disk.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a config if its output directory already exists.",
    )
    return parser.parse_args()


def save_dataset(repo_id: str, config: str, output_dir: Path, cache_dir: Path | None, num_proc: int | None) -> None:
    target_dir = output_dir / repo_to_dir_name(repo_id) / config
    if target_dir.exists():
        print(f"[exists] {target_dir}")

    print(f"[download] repo={repo_id} config={config}")
    dataset = load_dataset(
        repo_id,
        config,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[save] {target_dir}")
    dataset.save_to_disk(str(target_dir), num_proc=num_proc)


def write_manifest(output_dir: Path, datasets: list[str], configs: list[str]) -> None:
    manifest = output_dir / "CoIR-Retrieval_manifest.txt"
    lines = [
        "Local datasets downloaded from Hugging Face.",
        "",
        "Load one config with:",
        "  from datasets import load_from_disk",
        "  ds = load_from_disk('/path/to/CoIR-Retrieval__cosqa/default')",
        "",
        "Datasets:",
    ]
    for repo_id in datasets:
        lines.append(f"- {repo_id} -> {repo_to_dir_name(repo_id)}/")
        for config in configs:
            lines.append(f"  - {config}/")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for repo_id in args.datasets:
        for config in args.configs:
            target_dir = args.output_dir / repo_to_dir_name(repo_id) / config
            if args.skip_existing and target_dir.exists():
                print(f"[skip] {target_dir}")
                continue
            save_dataset(repo_id, config, args.output_dir, args.cache_dir, args.num_proc)

    write_manifest(args.output_dir, args.datasets, args.configs)
    print(f"[done] saved under {args.output_dir}")


if __name__ == "__main__":
    main()
