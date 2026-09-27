"""Convert the full LeCaRDv2 candidate collection to MTEB-style JSONL."""

import argparse
import json
import os
import tempfile
from pathlib import Path


TEXT_FIELDS = ("qw", "fact", "reason", "result")


def _load_candidate(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        candidate = json.load(source)

    pid = candidate.get("pid")
    if str(pid) != path.stem:
        raise ValueError(f"PID does not match filename: {path} (pid={pid!r})")
    for field in TEXT_FIELDS:
        if not isinstance(candidate.get(field), str):
            raise ValueError(f"{path}: {field!r} must be a string")
    charges = candidate.get("charge")
    if not isinstance(charges, list) or not all(
        isinstance(charge, str) for charge in charges
    ):
        raise ValueError(f"{path}: 'charge' must be a list of strings")
    return candidate


def _to_mteb_record(candidate: dict) -> dict:
    text = "\n".join(
        [
            candidate["qw"],
            candidate["fact"],
            candidate["reason"],
            candidate["result"],
            ",".join(candidate["charge"]),
        ]
    ) + "\n"
    return {"_id": str(candidate["pid"]), "title": "", "text": text}


def convert_corpus(source_dir: Path, output_path: Path) -> int:
    """Convert numerically named candidate JSON files to one corpus JSONL."""
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    if not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    try:
        source_files = sorted(source_dir.glob("*.json"), key=lambda path: int(path.stem))
    except ValueError as exc:
        raise ValueError(f"Candidate filenames must be numeric under: {source_dir}") from exc
    if not source_files:
        raise ValueError(f"No candidate JSON files found under: {source_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids = set()
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
                candidate = _load_candidate(source_path)
                doc_id = str(candidate["pid"])
                if doc_id in seen_ids:
                    raise ValueError(f"Duplicate candidate PID: {doc_id}")
                seen_ids.add(doc_id)
                output_file.write(json.dumps(_to_mteb_record(candidate)) + "\n")
        os.chmod(temp_path, 0o664)
        os.replace(temp_path, output_path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    return len(source_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert full LeCaRDv2 candidates to MTEB-style corpus JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, help="candidate_55192")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_corpus(args.input, args.output)
    print(f"Wrote {count} documents to {args.output}")


if __name__ == "__main__":
    main()
