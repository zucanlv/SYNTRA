import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter
from pathlib import Path

DATA_PATH = Path(
    "/data/share/project/shared_datasets/DSA/syn_data/RQ1-gen-ori/0606/gen/msmarco-gen-10k-0606.jsonl"
)

# ── helpers ───────────────────────────────────────────────────────────────

def stats_block(arr, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  样本数 (total):   {len(arr):>10,}")
    print(f"  均值   (mean):    {arr.mean():>10.2f}")
    print(f"  中位数 (median):  {np.median(arr):>10.2f}")
    print(f"  标准差 (std):     {arr.std():>10.2f}")
    print(f"  最小值 (min):     {arr.min():>10}")
    print(f"  最大值 (max):     {arr.max():>10}")
    for pct in (25, 75, 90, 95, 99):
        print(f"  P{pct:<2}:             {np.percentile(arr, pct):>10.1f}")

def bucket_distribution(arr, buckets, label, total):
    """Print count + percentage for each bucket defined by (lo, hi, name)."""
    print(f"\n── {label} 分布 ──")
    print(f"  {'区间':<22}  {'数量':>8}  {'占比':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}")
    for lo, hi, name in buckets:
        if hi is None:
            mask = arr >= lo
        else:
            mask = (arr >= lo) & (arr < hi)
        cnt = mask.sum()
        print(f"  {name:<22}  {cnt:>8,}  {cnt/total*100:>7.2f}%")

def collect_query_lengths(data_path):
    word_lengths = []
    char_lengths = []

    with data_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] line {line_no}: JSON decode error: {e}")
                continue
            query = record.get("query", "")
            if query is None:
                query = ""
            elif not isinstance(query, str):
                query = str(query)
            word_lengths.append(len(query.split()))
            char_lengths.append(len(query))

    return np.array(word_lengths), np.array(char_lengths)


def save_histogram(data_path, save_path, word_lengths, char_lengths, total):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Query Length Distribution\n"
        f"(n={total:,}  |  {data_path.name})",
        fontsize=13, y=1.01
    )

    ax = axes[0]
    w_min, w_max = int(word_lengths.min()), int(word_lengths.max())
    bins_w = np.arange(w_min, w_max + 2) - 0.5   # one bin per integer value
    counts_w, edges_w, patches_w = ax.hist(
        word_lengths, bins=bins_w, color="#4C8BE2", edgecolor="white", linewidth=0.6
    )
    ax.axvline(word_lengths.mean(),   color="#E25C4C", linewidth=1.8,
               linestyle="--", label=f"Mean={word_lengths.mean():.1f}")
    ax.axvline(np.median(word_lengths), color="#F5A623", linewidth=1.8,
               linestyle="-.",  label=f"Median={np.median(word_lengths):.0f}")
    ax.set_xlabel("Query Length (words)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Word Count Distribution", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for cnt, patch in zip(counts_w, patches_w):
        if cnt > 0:
            pct = cnt / total * 100
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                cnt + total * 0.003,
                f"{pct:.1f}%" if pct >= 1 else "",
                ha="center", va="bottom", fontsize=7, color="#333333"
            )
    ax.legend(fontsize=9)
    ax.set_xlim(w_min - 1, w_max + 1)

    ax = axes[1]
    bins_c = np.arange(0, char_lengths.max() + 12, 10)
    counts_c, edges_c, patches_c = ax.hist(
        char_lengths, bins=bins_c, color="#57B87A", edgecolor="white", linewidth=0.6
    )
    ax.axvline(char_lengths.mean(),     color="#E25C4C", linewidth=1.8,
               linestyle="--", label=f"Mean={char_lengths.mean():.1f}")
    ax.axvline(np.median(char_lengths), color="#F5A623", linewidth=1.8,
               linestyle="-.",  label=f"Median={np.median(char_lengths):.0f}")
    ax.set_xlabel("Query Length (characters)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Character Count Distribution", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for cnt, patch in zip(counts_c, patches_c):
        if cnt / total * 100 >= 1:
            pct = cnt / total * 100
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                cnt + total * 0.003,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=7, color="#333333"
            )
    ax.legend(fontsize=9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compile query length statistics for a training JSONL file."
    )
    parser.add_argument(
        "jsonl",
        nargs="?",
        type=Path,
        default=DATA_PATH,
        help=f"path to JSONL (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for the histogram PNG (default: next to JSONL)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="print statistics only; do not save the histogram PNG",
    )
    args = parser.parse_args()

    data_path = args.jsonl
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    word_lengths, char_lengths = collect_query_lengths(data_path)
    total = len(word_lengths)
    if total == 0:
        raise ValueError(f"No valid JSONL records found in: {data_path}")

    stats_block(word_lengths, "Query 长度（词数 / word count）")

    word_buckets = [
        (1,  3,  "1–2  词"),
        (3,  6,  "3–5  词"),
        (6,  11, "6–10 词"),
        (11, 16, "11–15 词"),
        (16, 21, "16–20 词"),
        (21, None, "≥21  词"),
    ]
    bucket_distribution(word_lengths, word_buckets, "按词数", total)

    stats_block(char_lengths, "Query 长度（字符数 / char count）")

    char_buckets = [
        (1,  20,  "1–19   字符"),
        (20, 40,  "20–39  字符"),
        (40, 60,  "40–59  字符"),
        (60, 80,  "60–79  字符"),
        (80, 100, "80–99  字符"),
        (100, None, "≥100   字符"),
    ]
    bucket_distribution(char_lengths, char_buckets, "按字符数", total)

    print("\n── 词数最频繁的 Top-15 ──")
    print(f"  {'词数':>6}  {'数量':>8}  {'占比':>8}")
    print(f"  {'------':>6}  {'--------':>8}  {'--------':>8}")
    for wc, cnt in Counter(word_lengths.tolist()).most_common(15):
        print(f"  {wc:>6}  {cnt:>8,}  {cnt/total*100:>7.2f}%")

    print(f"\n数据文件: {data_path}")
    print("脚本运行完毕。")

    if args.no_plot:
        return

    out_dir = args.output_dir if args.output_dir else data_path.parent
    save_path = out_dir / f"{data_path.stem}-query_length_distribution.png"
    save_histogram(data_path, save_path, word_lengths, char_lengths, total)
    print(f"\n直方图已保存至: {save_path}")


if __name__ == "__main__":
    main()
