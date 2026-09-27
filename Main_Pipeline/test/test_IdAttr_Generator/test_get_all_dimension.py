"""
从 IdAttr_Generator 输出的 JSON 中提取每个 passage 的 dimension 列表（不含 options），
便于分析各 passage 的维度分布。
"""

import json
from pathlib import Path


def extract_dimensions_from_file(input_path: str, output_path: str | None = None) -> list[list[str]]:
    """
    从 JSON 文件中提取每个 passage 的 dimension 列表（仅 dimensions，不含 passage）。
    每个 passage 的 dimensions 占一行（JSONL），便于分析。

    Args:
        input_path: 输入的 IdAttr_Generator 结果 JSON 路径
        output_path: 输出路径；若为 None 则与输入同目录，命名为 *_dimensions.jsonl

    Returns:
        列表，每项为对应 passage 的 dimension 列表
    """
    path = Path(input_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = []
    for item in data:
        attributes = item.get("attributes", [])
        dimensions = [a.get("dimension") for a in attributes if a.get("dimension")]
        result.append(dimensions)

    out = output_path or str(path.parent / f"{path.stem}_dimensions.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for dims in result:
            f.write(json.dumps(dims, ensure_ascii=False) + "\n")
    print(f"已提取 {len(result)} 个 passage 的 dimensions，保存至: {out}")
    return result


if __name__ == "__main__":
    # 默认处理当前常用的结果文件
    base = Path(__file__).resolve().parent.parent
    input_json = "/home/zclyu/Synthetic_Data/SyntheticDataResearch/Main_Pipeline/Results/IdAttr_Generator_20260202_094824.json"
    extract_dimensions_from_file(str(input_json))
