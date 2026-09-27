"""
Test: 将多条 passage 同时送入大模型，只得到 dimension 列表（不要求 option、identifier）。
python test_multiple_passage_for_dimension.py --limit 20 --passage-file ...
Usage:
  python test_multiple_passage_for_dimension.py [--passage-file PATH] [--limit N] [--num-passages M] [--out-format json|jsonl]

  --num-passages: 一次送入的 passage 数量。默认 None = 全部 passage 一起送入；设为整数时仅用该数量的 passage 做一次调用。
"""

import os
import json
import json5
import logging
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from datetime import datetime
import pathlib
import random

# Configure logging to write to both a file and the console
log_dir = pathlib.Path(__file__).resolve().parent.parent / "Logs"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"test_multiple_passage_dimension_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)


def _build_dimension_only_prompt(passages: list[str]) -> str:
    """Build prompt that asks for dimensions only (no options, no identifiers)."""
    numbered = "\n\n---\n\n".join(f"[Passage {i+1}]\n{p}" for i, p in enumerate(passages))
    return f"""
You are a professional data synthesis and NLP expert. Your task is to analyze the following passage(s) and output only the **query attribute dimensions** (dimension names) that are useful for diverse query generation across these passages.

- Query Attribute Dimension: a non-core feature dimension that controls how queries can be asked (e.g., question intent, reasoning depth, granularity, answer form). Dimensions should be independent and applicable to the passage(s) domain/topic.
- Brainstorm as many dimension names as possible that are independent and practical for query diversity. Output only the list of dimension names (no upper limit; aim for comprehensive coverage).

Input (multiple passages):
{numbered}

Output Format (MANDATORY, machine-parsable):
- Output MUST be a single valid JSON object (RFC 8259).
- Output MUST contain ONLY the JSON object: no Markdown, no code fences, no commentary, no extra keys, no trailing text.
- Use double quotes for all strings and keys. No trailing commas.

JSON schema (output MUST follow exactly):
{{
  "dimensions": ["dimension_name_1", "dimension_name_2", ...]
}}
"""


@register_tool('dimension_only_generator')
class DimensionOnlyGenerator(BaseTool):
    description = "Generate only dimension names for multiple passages (no options, no identifiers)."
    parameters = [{
        'name': 'passages',
        'type': 'array',
        'description': 'List of passage strings to analyze together.',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        params_dict = json5.loads(params)
        passages = params_dict.get('passages', [])
        if isinstance(passages, str):
            passages = [passages]

        user_prompt = _build_dimension_only_prompt(passages)
        llm_cfg = kwargs.get('llm_cfg', {
            'model': 'gpt-4o-mini',
            'model_server': os.getenv('OPENAI_BASE_URL'),
            'api_key': os.getenv('OPENAI_API_KEY'),
        })
        designer = Assistant(llm=llm_cfg)
        raw_output = ""
        for response in designer.run([{'role': 'user', 'content': user_prompt}]):
            raw_output = response[-1]['content']

        try:
            parsed = json5.loads(raw_output)
        except Exception as e:
            logging.warning(f"Failed to parse dimension JSON output: {e}")
            parsed = {}
        dimensions = parsed.get('dimensions', [])
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        return json5.dumps({
            'passages': passages,
            'dimensions': dimensions,
        }, ensure_ascii=False)

def _parse_args():
    import argparse
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Test: 多条 passage 同时送入大模型，只得到 dimension 列表（不要求 option、identifier）。"
    )
    p.add_argument("--passage-file", type=pathlib.Path,
                   default=base_dir / "Results" / "diverse_passages_20260201_113527.jsonl",
                   help="Input: JSON array of objects with 'passage' key, or JSONL (one JSON per line with 'passage').")
    p.add_argument("--out-format", choices=["jsonl", "json"], default="json",
                   help="Output format: json or jsonl.")
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of passages to load from file (default: all).")
    p.add_argument("--num-passages", type=int, default=None,
                   help="一次送入大模型的 passage 数量。默认 None = 全部 passage 一起送入；设为整数时仅用该数量的 passage 做一次调用。")
    p.add_argument("--seed", type=int, default=None, help="Random seed for sampling when using --limit.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    llm_cfg = {
        'model': 'gpt-4o-mini',
        'model_server': os.getenv('OPENAI_BASE_URL'),
        'api_key': os.getenv('OPENAI_API_KEY'),
    }

    tool = DimensionOnlyGenerator()
    passage_file = args.passage_file

    if not passage_file.exists():
        logging.error(f"Passage file not found: {passage_file}")
    else:
        logging.info(f"Loading passages from {passage_file}")
        with open(passage_file, 'r', encoding='utf-8') as f:
            raw = f.read()

        passages = []
        # Try whole-file JSON first (single array or list of objects with 'passage')
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'passage' in item:
                        passages.append(item.get('passage', ''))
                    elif isinstance(item, dict):
                        passages.append(item.get('passage', ''))
            elif isinstance(data, dict) and 'passage' in data:
                passages = [data.get('passage', '')]
            elif isinstance(data, dict):
                # e.g. {"passages": [...]}
                for key in ('passages', 'items', 'data'):
                    if key in data and isinstance(data[key], list):
                        for item in data[key]:
                            if isinstance(item, dict):
                                passages.append(item.get('passage', ''))
                            elif isinstance(item, str):
                                passages.append(item)
                        break
        except json.JSONDecodeError:
            pass

        # Fallback: JSONL (one JSON object per line)
        if not passages:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            total = len(lines)
            if args.limit is not None:
                if args.seed is not None:
                    random.seed(args.seed)
                lines = random.sample(lines, min(args.limit, total)) if total else []
            for line in lines:
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        passages.append(data.get('passage', ''))
                except Exception as e:
                    logging.warning(f"Skip line: {e}")

        if args.limit is not None and len(passages) > args.limit:
            if args.seed is not None:
                random.seed(args.seed)
            passages = random.sample(passages, args.limit)

        # 决定本次调用用多少条 passage：默认全部，或 --num-passages 指定数量
        n_use = len(passages) if args.num_passages is None else min(args.num_passages, len(passages))
        batch = passages[:n_use]
        logging.info(f"Sending {len(batch)} passage(s) in one call (num_passages={args.num_passages}, total loaded={len(passages)}).")

        result_str = tool.call(json5.dumps({'passages': batch}), llm_cfg=llm_cfg)
        result = json5.loads(result_str)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = base_dir / "Results"
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.out_format == "jsonl":
            out_path = out_dir / f"IdAttr_Generator_dimensions_{ts}.jsonl"
            with open(out_path, 'w', encoding='utf-8') as out_f:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            out_path = out_dir / f"IdAttr_Generator_dimensions_{ts}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        logging.info(f"Completed. Dimensions: {result.get('dimensions', [])}. Results saved to {out_path}.")
