"""
Script: 4-21-count-training-data-pos-neg.py

Stream a training JSONL file (fields: prompt, query, pos, neg) and report:
  - number of rows (non-empty lines)
  - total count of pos / neg items across all rows
  - average pos / neg per row
  - optional per-row min/max for pos and neg lengths
  - per-row length histograms (bucketed) for pos and neg
  - optional PNG export (matplotlib) with the same bucketing as the terminal chart

Usage:
  python 4-21-count-training-data-pos-neg.py
  python 4-21-count-training-data-pos-neg.py /path/to/dataset.jsonl
  python 4-21-count-training-data-pos-neg.py data.jsonl --bin-width 2
  python /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/corpus_preprocess/4-21-count-training-data-pos-neg.py /data/share/project/shared_datasets/DSA/syn_data/msmarco/title-msmarco-10k/title-msmarco-74k-ori-v1-10k.jsonl --save-png /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/corpus_preprocess/count-training-data-pos-neg/title-msmarco-74k-ori-v1-10k.PNG
  python 4-21-count-training-data-pos-neg.py data.jsonl --save-png   # auto path next to JSONL
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SAVE_PNG_NEXT_TO_INPUT = object()

DEFAULT_JSONL = Path(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ2-general-retrieve-task/0606-touche/touche-10k-0606.jsonl"
)


def _list_len(field: object, field_name: str, line_no: int) -> int:
    if field is None:
        return 0
    if isinstance(field, list):
        return len(field)
    print(
        f"  [warn] line {line_no}: '{field_name}' is not a list "
        f"(got {type(field).__name__}), counting as 0",
        file=sys.stderr,
    )
    return 0


def _bucketed_counts(hist: Counter[int], bin_width: int) -> list[tuple[int, int, int]]:
    """Return sorted [(low, high_inclusive, count), ...] after merging by bin_width."""
    if bin_width < 1:
        raise ValueError("bin_width must be >= 1")
    if bin_width == 1:
        return [(k, k, hist[k]) for k in sorted(hist)]
    merged: dict[int, int] = defaultdict(int)
    for k, c in hist.items():
        lo = (k // bin_width) * bin_width
        merged[lo] += c
    hi_off = bin_width - 1
    return [(lo, lo + hi_off, merged[lo]) for lo in sorted(merged)]


def _print_histogram(
    title: str,
    hist: Counter[int],
    num_rows: int,
    bin_width: int,
    bar_chars: int,
) -> None:
    rows = _bucketed_counts(hist, bin_width)
    if not rows:
        print(f"\n  [{title}] (no data)")
        return
    max_cnt = max(c for _, _, c in rows)
    scale = max_cnt if max_cnt > 0 else 1

    print(f"\n  [{title}]  (rows={num_rows:,}, bin_width={bin_width})")
    label_w = max(len(f"{lo}-{hi}") for lo, hi, _ in rows)
    for lo, hi, cnt in rows:
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        bar_len = max(1, int(round(bar_chars * cnt / scale))) if max_cnt > 0 else 0
        if cnt > 0 and bar_len == 0:
            bar_len = 1
        bar = "█" * bar_len
        pct = 100.0 * cnt / num_rows if num_rows else 0.0
        print(f"    {label:>{label_w}s}  {bar:<{bar_chars}s}  {cnt:>6,}  ({pct:5.1f}%)")


def _bucket_labels(rows: list[tuple[int, int, int]]) -> list[str]:
    return [f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi, _ in rows]


def _save_histogram_png(
    out_path: Path,
    jsonl_path: Path,
    pos_hist: Counter[int],
    neg_hist: Counter[int],
    num_rows: int,
    bin_width: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos_rows = _bucketed_counts(pos_hist, bin_width)
    neg_rows = _bucketed_counts(neg_hist, bin_width)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7.2),
        dpi=140,
        constrained_layout=True,
    )
    fig.suptitle(
        f"{jsonl_path.name}\nrows={num_rows:,} · bin_width={bin_width}",
        fontsize=11,
    )

    for ax, rows, name, color in (
        (axes[0], pos_rows, "pos length / row", "#2e7d32"),
        (axes[1], neg_rows, "neg length / row", "#1565c0"),
    ):
        if not rows:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            continue
        labels = _bucket_labels(rows)
        counts = [c for *_, c in rows]
        x = range(len(labels))
        ax.bar(x, counts, color=color, edgecolor="white", linewidth=0.7)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=9)
        ax.set_ylabel("number of rows")
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.38)

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count pos/neg list lengths in a training JSONL file."
    )
    parser.add_argument(
        "jsonl",
        nargs="?",
        type=Path,
        default=DEFAULT_JSONL,
        help=f"path to JSONL (default: {DEFAULT_JSONL})",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=1,
        metavar="N",
        help="merge length buckets: each bin covers N consecutive counts (default: 1)",
    )
    parser.add_argument(
        "--bar-width",
        type=int,
        default=36,
        metavar="W",
        help="max characters for the bar in histogram (default: 36)",
    )
    parser.add_argument(
        "--save-png",
        nargs="?",
        type=Path,
        const=_SAVE_PNG_NEXT_TO_INPUT,
        default=None,
        metavar="PATH",
        help=(
            "save matplotlib histogram (pos + neg) to PNG; "
            "omit PATH to write <jsonl_stem>_posneg_bin<N>.png next to the JSONL; "
            "if PATH is a directory, the file is created inside it"
        ),
    )
    args = parser.parse_args()
    jsonl_path: Path = args.jsonl

    if not jsonl_path.is_file():
        print(f"Error: file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    num_rows = 0
    total_pos = 0
    total_neg = 0
    parse_errors = 0
    min_pos = min_neg = None
    max_pos = max_neg = 0
    pos_hist: Counter[int] = Counter()
    neg_hist: Counter[int] = Counter()

    print(f"Reading: {jsonl_path}\n")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                parse_errors += 1
                print(f"  [warn] line {line_no}: JSON decode error: {e}", file=sys.stderr)
                continue

            n_pos = _list_len(entry.get("pos"), "pos", line_no)
            n_neg = _list_len(entry.get("neg"), "neg", line_no)

            num_rows += 1
            total_pos += n_pos
            total_neg += n_neg

            if min_pos is None:
                min_pos, min_neg = n_pos, n_neg
            else:
                min_pos = min(min_pos, n_pos)
                min_neg = min(min_neg, n_neg)
            max_pos = max(max_pos, n_pos)
            max_neg = max(max_neg, n_neg)
            pos_hist[n_pos] += 1
            neg_hist[n_neg] += 1

    if num_rows == 0:
        print("No data rows found (file empty or only blank lines).")
        if parse_errors:
            print(f"  JSON parse errors: {parse_errors}", file=sys.stderr)
        sys.exit(1)

    avg_pos = total_pos / num_rows
    avg_neg = total_neg / num_rows

    sep = "─" * 52
    print(sep)
    print(f"  File                      : {jsonl_path}")
    print(f"  Rows (JSON objects)       : {num_rows:,}")
    print(f"  Total pos items         : {total_pos:,}")
    print(f"  Total neg items         : {total_neg:,}")
    print(f"  Avg pos per row         : {avg_pos:.4f}")
    print(f"  Avg neg per row         : {avg_neg:.4f}")
    print(f"  Min pos / max pos       : {min_pos} / {max_pos}")
    print(f"  Min neg / max neg       : {min_neg} / {max_neg}")
    if parse_errors:
        print(f"  Skipped (parse errors)  : {parse_errors:,}")
    print(sep)

    bw = args.bin_width
    if bw < 1:
        print("Error: --bin-width must be >= 1", file=sys.stderr)
        sys.exit(1)
    bar_w = max(8, args.bar_width)

    _print_histogram("pos length / row", pos_hist, num_rows, bw, bar_w)
    _print_histogram("neg length / row", neg_hist, num_rows, bw, bar_w)
    print()

    png_arg = args.save_png
    if png_arg is not None:
        if png_arg is _SAVE_PNG_NEXT_TO_INPUT:
            out_png = jsonl_path.parent / f"{jsonl_path.stem}_posneg_bin{bw}.png"
        else:
            out_png = png_arg
        if out_png.exists() and out_png.is_dir():
            out_png = out_png / f"{jsonl_path.stem}_posneg_bin{bw}.png"
        try:
            _save_histogram_png(out_png, jsonl_path, pos_hist, neg_hist, num_rows, bw)
        except ImportError as e:
            print(f"Error: matplotlib is required for --save-png ({e})", file=sys.stderr)
            sys.exit(1)
        print(f"Histogram PNG saved to: {out_png}")


if __name__ == "__main__":
    main()
