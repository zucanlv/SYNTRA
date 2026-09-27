#!/usr/bin/env python3
"""Remove the top-level ``prompt`` field from every record in a JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 JSONL 每一行中的顶层 prompt 字段删除并写入新文件。"
    )
    parser.add_argument("input", type=Path, help="源 JSONL 文件路径")
    parser.add_argument("output", type=Path, help="输出 JSONL 文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if input_path == output_path:
        raise SystemExit("错误：输出文件不能与源文件相同。")
    if not input_path.is_file():
        raise SystemExit(f"错误：源文件不存在或不是普通文件：{input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"错误：第 {line_number} 行不是合法 JSON：{exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise SystemExit(f"错误：第 {line_number} 行的 JSON 顶层必须是对象。")

            record.pop("prompt", None)
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()