#!/usr/bin/env python3
"""Package codetrans-dl JSONL sources and winning 4q training data."""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


SOURCE_FILES = {
    "source/strong_2q/codetrans-dl-2q-query1-20260811.jsonl":
        "codetrans-dl-2q-query1-20260811.jsonl",
    "source/strong_2q/codetrans-dl-2q-query1-20260811_exclude_other_pos_save_non_pos.jsonl":
        "codetrans-dl-2q-query1-20260811_exclude_other_pos_save_non_pos.jsonl",
    "source/strong_2q/codetrans-dl-2q-query2-20260811.jsonl":
        "codetrans-dl-2q-query2-20260811.jsonl",
    "source/strong_2q/codetrans-dl-2q-query2-20260811_exclude_other_pos_save_non_pos.jsonl":
        "codetrans-dl-2q-query2-20260811_exclude_other_pos_save_non_pos.jsonl",
    **{
        f"source/generated_4q/{name}": name
        for query_index in range(1, 5)
        for name in (
            f"codetrans-dl-4q-query{query_index}-20260812.jsonl",
            f"codetrans-dl-4q-query{query_index}-20260812_"
            "exclude_other_pos_save_non_pos.jsonl",
        )
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--winner-mix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--sample-per-query", type=int, default=100)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    for relative_path, source_name in SOURCE_FILES.items():
        source = args.source_root / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = args.output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    sampled: dict[int, list[dict]] = {}
    for query_index in (3, 4):
        source = args.source_root / (
            f"codetrans-dl-4q-query{query_index}-20260812_"
            "exclude_other_pos_save_non_pos.jsonl"
        )
        records = load_jsonl(source)
        sampled[query_index] = random.Random(args.seed + query_index).sample(
            records, args.sample_per_query
        )
        write_jsonl(
            args.output / "winner" /
            f"codetrans-dl-q{query_index}-filtered-sample{args.sample_per_query}.jsonl",
            sampled[query_index],
        )

    rebuilt_mix = sampled[3] + sampled[4]
    random.Random(args.seed).shuffle(rebuilt_mix)
    winner_mix = load_jsonl(args.winner_mix)
    if rebuilt_mix != winner_mix:
        raise ValueError("rebuilt winner mix does not match the trained JSONL")
    shutil.copy2(
        args.winner_mix,
        args.output / "winner" / "codetrans-dl-q3q4-filtered-sample200.jsonl",
    )

    manifest = {
        "dataset": "codetrans-dl",
        "release": "4q-20260823",
        "winning_ndcg_at_10": 0.38374,
        "baseline_2q_ndcg_at_10": 0.37961,
        "sample_seed": args.seed,
        "files": [],
    }
    for path in sorted(args.output.rglob("*.jsonl")):
        manifest["files"].append(
            {
                "path": str(path.relative_to(args.output)),
                "rows": line_count(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with (args.output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    readme = f"""# codetrans-dl 4q release data

This bundle contains all strong-2q and generated-4q source JSONL files used
in the 2026-08-23 experiments.

The winning run (NDCG@10 = 0.38374) used these five training inputs, in order:

1. source/strong_2q/codetrans-dl-2q-query2-20260811.jsonl
2. source/strong_2q/codetrans-dl-2q-query2-20260811_exclude_other_pos_save_non_pos.jsonl
3. source/strong_2q/codetrans-dl-2q-query1-20260811.jsonl
4. source/strong_2q/codetrans-dl-2q-query1-20260811_exclude_other_pos_save_non_pos.jsonl
5. winner/codetrans-dl-q3q4-filtered-sample200.jsonl

The winner file contains {args.sample_per_query} q3 and
{args.sample_per_query} q4 rows. Sampling is deterministic with seed
{args.seed}; the separate component files are included under winner/.

`exclude_other_pos_save_non_pos` keeps every query. When the query's origin
document appears in `pos`, it retains only that origin document in `pos`.
Queries without a mapping, or whose origin document is absent from `pos`,
remain unchanged. Negative passages are not modified.

See manifest.json for row counts and SHA-256 checksums.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
