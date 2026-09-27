from __future__ import annotations

import csv
from pathlib import Path

from .prepare_tasks import TASK_ALIASES, norm_task


MTEB_SUPPORTED_NAMES = {
    "MedicalQARetrieval",
    "CmedqaRetrieval",
    "MedicalRetrieval",
    "ChemNQRetrieval",
}

MTEB_SUPPORTED_KEYWORDS = (
    "MTEB Retrieval",
    "MTEB风格",
    "C-MTEB Retrieval",
    "BEIR格式",
    "BEIR标准",
    "BEIR版",
    "BEIR/DPR格式",
    "TREC run格式",
)


def is_mteb_or_beir_supported(row: dict[str, str]) -> bool:
    name = row["数据集名称"]
    org = row.get("组织形式", "")
    if name in MTEB_SUPPORTED_NAMES:
        return True
    if name.startswith("BRIGHT-"):
        return True
    return any(keyword in org for keyword in MTEB_SUPPORTED_KEYWORDS)


def has_local_adapter(row: dict[str, str]) -> bool:
    return norm_task(row["数据集名称"]) in TASK_ALIASES


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Print task support status from benchmark.csv.")
    parser.add_argument("--csv", type=Path, default=Path("benchmark.csv"))
    args = parser.parse_args()

    with args.csv.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print("MTEB/BEIR-supported tasks:")
    for row in rows:
        if is_mteb_or_beir_supported(row):
            print(f"  - {row['数据集名称']}")

    print("\nNon-MTEB tasks supported by local adapters:")
    for row in rows:
        if not is_mteb_or_beir_supported(row) and has_local_adapter(row):
            print(f"  - {row['数据集名称']}")

    missing = [row["数据集名称"] for row in rows if not is_mteb_or_beir_supported(row) and not has_local_adapter(row)]
    print("\nNon-MTEB tasks missing local adapters:")
    if missing:
        for name in missing:
            print(f"  - {name}")
    else:
        print("  - none")


if __name__ == "__main__":
    main()
