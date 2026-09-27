"""
Adapted from https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/evaluation_arguments.py
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalArgs:
    # model loading
    model_name_or_path: str = field(
        metadata={"help": "The model name or path.", "required": True}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "whether to trust remote code when loading the model"}
    )
    use_fp16: bool = field(
        default=True, metadata={"help": "whether to use fp16 for inference"}
    )
    use_bf16: bool = field(
        default=False, metadata={"help": "whether to use bf16 for inference"}
    )
    prompt_template: str = field(
        default="Instruct: {}\nQuery: {}", metadata={"help": "Format for query instruction"}
    )
    assert_prompts_exist: bool = field(
        default=True, metadata={"help": "whether to assert that prompts exist for all tasks"}
    )
    
    # model inference
    normalize_embeddings: bool = field(
        default=True, metadata={"help": "whether to normalize the embeddings"}
    )
    batch_size: int = field(
        default=32, metadata={"help": "Batch size for inference"}
    )
    max_length: int = field(
        default=512, metadata={"help": "Maximum sequence length for the model"}
    )
    
    
    # evluation
    benchmark_name: str = field(
        default="MTEB(Multilingual, v2)", metadata={"help": "The benchmark name to evaluate on."}
    )
    results_output_folder: str = field(
        default="./mteb_results", metadata={"help": "The path to save evaluation results."}
    )
    only_download_data: bool = field(
        default=False, metadata={"help": "Whether to only download the benchmark data without running evaluation."}
    )

    def __post_init__(self):
        # replace "\\n" with "\n"
        if "\\n" in self.prompt_template:
            self.prompt_template = self.prompt_template.replace("\\n", "\n")
