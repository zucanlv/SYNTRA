"""
根据 pid 从 FAISS 索引相关数据中取出对应的 passage。
与 test_msmarco.py 一致：使用同一目录下的 doc_dict.pkl（pid -> passage），无需加载 FAISS 索引本身。

用法:
  python test_get_passage_from_pid.py --pids 1524580 3448753 5238655
  python test_get_passage_from_pid.py --pids 1524580 --faiss-dir /path/to/faiss/ms_marco
"""

import argparse
import os
import pickle

# 与 test_msmarco.py 一致的默认路径
DEFAULT_FAISS_DIR = "/data/share/project/zucan/data/faiss/ms_marco"


def load_doc_dict(faiss_dir):
    """仅加载 doc_dict.pkl（pid -> passage）。"""
    doc_dict_path = os.path.join(faiss_dir, "doc_dict.pkl")
    if not os.path.exists(doc_dict_path):
        raise FileNotFoundError(f"未找到 doc_dict.pkl：{doc_dict_path}")
    with open(doc_dict_path, "rb") as f:
        return pickle.load(f)


def _extract_text(entry):
    """Extract passage text from a doc_dict entry (dict or legacy string)."""
    if isinstance(entry, dict):
        return entry.get("text", "")
    if isinstance(entry, str):
        return entry
    return ""


def get_passages_by_pids(pids, faiss_dir=DEFAULT_FAISS_DIR):
    """
    根据 pid 列表取出对应 passage。
    pids: list of int 或 str
    返回: list of dict [{"pid": pid, "passage": str}, ...]，未找到的 pid 的 passage 为 ""。
    """
    doc_dict = load_doc_dict(faiss_dir)
    results = []
    for pid in pids:
        entry = doc_dict.get(pid)
        if entry is None and str(pid).isdigit():
            entry = doc_dict.get(str(pid)) or doc_dict.get(int(pid))
        results.append({"pid": pid, "passage": _extract_text(entry) if entry else ""})
    return results


def main():
    parser = argparse.ArgumentParser(description="根据 pid 从 doc_dict 取出 passage")
    parser.add_argument("--pids", type=str, nargs="+", required=True, help="一个或多个 passage id，例如: --pids 1524580 3448753")
    parser.add_argument("--faiss-dir", type=str, default=DEFAULT_FAISS_DIR, help="FAISS 数据目录（含 doc_dict.pkl）")
    parser.add_argument("--max-len", type=int, default=200, help="打印时 passage 截断长度，0 表示不截断")
    args = parser.parse_args()

    # 将传入的 pid 转为 int（若为数字字符串）
    pids = []
    for p in args.pids:
        try:
            pids.append(int(p))
        except ValueError:
            pids.append(p)

    print("加载 doc_dict...")
    results = get_passages_by_pids(pids, faiss_dir=args.faiss_dir)

    print(f"\n共 {len(results)} 个 pid 的 passage:\n")
    for r in results:
        pid, passage = r["pid"], r["passage"]
        if args.max_len and len(passage) > args.max_len:
            display = passage[: args.max_len] + "..."
        else:
            display = passage or "(未找到)"
        print(f"  pid={pid}")
        print(f"    passage: {display}")
        print()


if __name__ == "__main__":
    main()

'''
  pid=1524580
    passage: Ho Chi Minh Creates Provisional Government: Following the surrender of Japan to Allied forces, Ho Chi Minh and his People's Congress create the National Liberation Committee of Vietnam to form a provisional government. Japan transfers all power to Ho's Vietminh. Ho Declares Independence of Vietnam. British Forces Land in Saigon, Return Authority to French. First American Dies in Vietnam: Lt. Col. A. Peter Dewey, head of American OSS mission, was killed by Vietminh troops while driving a jeep to the airport.

  pid=3448753
    passage: Ho Chi Minh Creates Provisional Government: Following the surrender of Japan to Allied forces, Ho Chi Minh and his People's Congress create the National Liberation Committee of Vietnam to form a provisional government. Japan transfers all power to Ho's Vietminh. Ho Declares Independence of Vietnam.

  pid=5238655
    passage: Ho Declares Independence of Vietnam. British Forces Land in Saigon, Return Authority to French. First American Dies in Vietnam: Lt. Col. A. Peter Dewey, head of American OSS mission, was killed by Vietminh troops while driving a jeep to the airport. Reports later indicated that his death was due to a case of mistaken identity -- he had been mistaken for a Frenchman.


'''