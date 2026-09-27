"""Convert the LegalBench-RAG raw-text corpus to pipeline JSONL.

Each source ``.txt`` document becomes one JSON object with:

    {"id": "<relative/source/path>.txt", "text": "<document text>"}

Relative paths preserve the upstream corpus identity and remain unique across
the ContractNLI, CUAD, MAUD, and PrivacyQA subsets.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path


def convert_corpus(source_dir: Path, output_path: Path) -> int:
    """Convert all non-empty ``.txt`` files below *source_dir* to JSONL."""
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    if not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    source_files = sorted(source_dir.rglob("*.txt"))
    if not source_files:
        raise ValueError(f"No .txt corpus files found under: {source_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            for source_path in source_files:
                text = source_path.read_text(encoding="utf-8").strip()
                if not text:
                    raise ValueError(f"Empty corpus document: {source_path}")
                record = {
                    "id": source_path.relative_to(source_dir).as_posix(),
                    "text": text,
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temp_path, output_path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    return len(source_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LegalBench-RAG raw .txt documents to corpus JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, help="Raw corpus root")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_corpus(args.input, args.output)
    print(f"Wrote {count} documents to {args.output}")


if __name__ == "__main__":
    main()
