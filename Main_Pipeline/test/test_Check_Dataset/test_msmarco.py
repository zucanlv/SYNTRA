
"""
简单检索脚本：对单个 query 在 MS MARCO FAISS 索引上检索 top-K 的 pid 与 passage。
数据集与 Passage_Diverse_Selection.py 一致：FAISS 索引 + id_map.pkl + doc_dict.pkl，编码模型 BAAI/bge-m3。

用法:
  python simple_retrieve.py --faiss-dir /path/to/faiss/ms_marco --top-k 20
  python simple_retrieve.py --query "your query here" --top-k 20
"""

import argparse
import os
import pickle
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# 默认 query（与 triplets_view.json 中一致）
DEFAULT_QUERY = (
    "Ho Chi Minh Creates Provisional Government: Following the surrender of Japan to Allied forces, "
    "Ho Chi Minh and his People's Congress create the National Liberation Committee of Vietnam to form a provisional government. "
    "Japan transfers all power to Ho's Vietminh. Ho Declares Independence of Vietnam. "
    "British Forces Land in Saigon, Return Authority to French. First American Dies in Vietnam: "
    "Lt. Col. A. Peter Dewey, head of American OSS mission, was killed by Vietminh troops while driving a jeep to the airport."
)

# 与 Passage_Diverse_Selection.py 一致的默认路径与模型
DEFAULT_FAISS_DIR = "/data/share/project/zucan/data/faiss/ms_marco"
MODEL_NAME = "BAAI/bge-m3"


def load_resources(faiss_dir, use_gpu=False):
    """加载 FAISS 索引、id_map、doc_dict。"""
    index_path = os.path.join(faiss_dir, "index.faiss")
    id_map_path = os.path.join(faiss_dir, "id_map.pkl")
    doc_dict_path = os.path.join(faiss_dir, "doc_dict.pkl")

    for p, name in [(index_path, "index.faiss"), (id_map_path, "id_map.pkl"), (doc_dict_path, "doc_dict.pkl")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到 {name}：{p}")

    index = faiss.read_index(index_path)
    if use_gpu and torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print(f"FAISS 使用 GPU 失败，使用 CPU: {e}")

    with open(id_map_path, "rb") as f:
        id_map = pickle.load(f)
    with open(doc_dict_path, "rb") as f:
        doc_dict = pickle.load(f)

    return index, id_map, doc_dict


def retrieve_top_k(query, faiss_dir=DEFAULT_FAISS_DIR, top_k=20, model_name=MODEL_NAME, device=None, use_gpu_faiss=False):
    """
    对单个 query 检索 top_k 个 passage。
    返回: list of dict, 每项为 {"rank": int, "pid": str, "passage": str, "score": float}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    index, id_map, doc_dict = load_resources(faiss_dir, use_gpu=use_gpu_faiss)
    model = SentenceTransformer(model_name, device=device)

    # 编码 query，L2 归一化（与建索引时一致，内积即余弦相似度）
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)

    # 检索 top_k
    scores, indices = index.search(q_emb, top_k)
    scores = scores[0]
    indices = indices[0]

    results = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        if idx < 0:
            continue
        pid = id_map[idx]
        entry = doc_dict.get(pid, {})
        passage = entry.get("text", "") if isinstance(entry, dict) else (entry or "")
        results.append({"rank": rank, "pid": pid, "passage": passage, "score": float(score)})

    return results


def main():
    parser = argparse.ArgumentParser(description="单 query 检索 top-K passage（pid + passage）")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="检索 query，默认使用脚本内预设的 Ho Chi Minh 相关 query")
    parser.add_argument("--faiss-dir", type=str, default=DEFAULT_FAISS_DIR, help="FAISS 索引目录（含 index.faiss, id_map.pkl, doc_dict.pkl）")
    parser.add_argument("--top-k", type=int, default=20, help="返回 top 数量，默认 20")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="编码模型")
    parser.add_argument("--faiss-gpu", action="store_true", help="FAISS 使用 GPU")
    parser.add_argument("--output", type=str, default=None, help="可选：将结果写入 JSON 文件")
    args = parser.parse_args()

    print("加载索引与模型...")
    results = retrieve_top_k(
        query=args.query,
        faiss_dir=args.faiss_dir,
        top_k=args.top_k,
        model_name=args.model_name,
        use_gpu_faiss=args.faiss_gpu,
    )

    print(f"\nQuery:\n  {args.query[:200]}...\n")
    print(f"Top {len(results)} 检索结果 (pid + passage):\n")
    for r in results:
        print(f"  [Rank {r['rank']}] pid={r['pid']} score={r['score']:.4f}")
        p = r["passage"]
        print(f"    passage: {p[:200] + '...' if len(p) > 200 else p}")
        print()

    if args.output:
        import json
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_data = {"query": args.query, "top_k": args.top_k, "results": results}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"结果已写入: {args.output}")

    return results


if __name__ == "__main__":
    main()


'''
Query:
  Ho Chi Minh Creates Provisional Government: Following the surrender of Japan to Allied forces, Ho Chi Minh and his People's Congress create the National Liberation Committee of Vietnam to form a provi...

Top 20 检索结果 (pid + passage):

  [Rank 1] pid=1524580 score=1.0000
    passage: Ho Chi Minh Creates Provisional Government: Following the surrender of Japan to Allied forces, Ho Chi Minh and his People's Congress create the National Liberation Committee of Vietnam to form a provi...

  [Rank 2] pid=3448753 score=0.8705
    passage: Ho Chi Minh Creates Provisional Government: Following the surrender of Japan to Allied forces, Ho Chi Minh and his People's Congress create the National Liberation Committee of Vietnam to form a provi...

  [Rank 3] pid=5238655 score=0.8437
    passage: Ho Declares Independence of Vietnam. British Forces Land in Saigon, Return Authority to French. First American Dies in Vietnam: Lt. Col. A. Peter Dewey, head of American OSS mission, was killed by Vie...

  [Rank 4] pid=4601990 score=0.8311
    passage: Ho Chi Minh Creates Provisional Government: Following the surrender of Japan to Allied forces, Ho Chi Minh and his People's Congress create the National Liberation Committee of Vietnam to form a provi...

  [Rank 5] pid=3857398 score=0.7751
    passage: British Forces Land in Saigon, Return Authority to French. First American Dies in Vietnam: Lt. Col. A. Peter Dewey, head of American OSS mission, was killed by Vietminh troops while driving a jeep to ...

  [Rank 6] pid=8322852 score=0.7073
    passage: Making the world better, one answer at a time. Before the Japanese occupation of Vietnam during the Second World War, the country had been part of the French Empire. When the Japanese surrendered, Fra...

  [Rank 7] pid=1803440 score=0.6959
    passage: There, he organized a Vietnamese guerrilla organizationâthe Viet Minh âto fight for Vietnamese independence. Japan occupied French Indochina in 1940 and collaborated with French officials loyal to...

  [Rank 8] pid=2140722 score=0.6884
    passage: 1945 At the close of World War II, Ho Chi Minh organized the Viet Minh to foment a large-scale uprising in Vietnam. The Viet Minh captured major cities across Vietnam and declared Vietnam an independe...

  [Rank 9] pid=4195148 score=0.6883
    passage: The British and Chinese accepted the surrender of the Japanese in Vietnam and the French re-entered the area and took over control again. Ho Chi Minh, a Vietnamese Communist, returned to Vietnam from ...

  [Rank 10] pid=4195154 score=0.6863
    passage: Ho Chi Minh died on September 2, 1969, 25 years after declaring Vietnamâs independence from France and nearly six years before his forces succeeded in reuniting North and South Vietnam under communi...

  [Rank 11] pid=2876969 score=0.6848
    passage: Before World War Two, Vietnam had been part of the French Empire. During the war, the country had been overrun by the Japanese. When the Japanese retreated, the people of Vietnam took the opportunity ...

  [Rank 12] pid=5173277 score=0.6805
    passage: Declaration of independence and national leadership. In 1945, Ho Chi Minh, leader of the communist Viet Minh organization, declared Vietnam s independence from Japan, in a speech that invoked the U.S....

  [Rank 13] pid=2743287 score=0.6721
    passage: Ho Chi Minh Comes Home. Once Ho was back in Vietnam, he established a headquarters in a cave in northern Vietnam and established the Viet Minh, whose goal was to rid Vietnam of the French and Japanese...

  [Rank 14] pid=46447 score=0.6696
    passage: Hours after Japanâs surrender in World War II, Vietnamese communist Ho Chi Minh declares the independence of Vietnam from France.

  [Rank 15] pid=3921006 score=0.6675
    passage: Ho Chi Minh died on September 2, 1969, 25 years after declaring Vietnamâs independence from France and nearly six years before his forces succeeded in reuniting North and South Vietnam under communi...

  [Rank 16] pid=274875 score=0.6637
    passage: In Vietnam, the Americans actually fought â therefore in the Cold War â, the USSR could not. However, to support the Communist cause, the Soviet Union armed its fellow Communist state, Chin...

  [Rank 17] pid=8682792 score=0.6625
    passage: Prior to World War II Vietnam had been a colony of the French. During World War II the Japanese took control of the area. When the war ended there was a power vacuum. Vietnamese revolutionary and comm...

  [Rank 18] pid=2566447 score=0.6616
    passage: Inspired by Chinese and Soviet communism, Ho Chi Minh formed the Viet Minh, or the League for the Independence of Vietnam, to fight both Japan and the French colonial administration. Japan withdrew it...

  [Rank 19] pid=4195150 score=0.6575
    passage: At the warâs end in 1945, Ho Chi Minh, leader of the communist Viet Minh organization, declared Vietnamâs independence in a speech that invoked the U.S. Declaration of Independence and the French ...

  [Rank 20] pid=3162320 score=0.6575
    passage: In early October 80,000 French troops arrived at Saigon with orders from the chairman of France's provisional government, Charles de Gaulle, to stay in the southern half of Vietnam. The French tried t...
'''