import mteb
import torch
from sentence_transformers import SentenceTransformer
from typing import Optional


# MultiGPUModel: from https://github.com/embeddings-benchmark/mteb/issues/27
class MultiGPUModel:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.gpu_pool = self.model.start_multi_process_pool()

    def encode(self, sentences, **kwargs):
        return self.model.encode_multi_process(sentences, self.gpu_pool, **kwargs)


class SentenceTransformerEncoder:
    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = False,
        use_fp16: bool = True,
        use_bf16: bool = False,
        normalize_embeddings: bool = True,
        max_length: int = 512,
        batch_size: int = 256,
        prompt_template: Optional[str] = None,
        prompts_dict: Optional[dict] = None,
        assert_prompts_exist: bool = True,
        only_download_data: bool = False,
        **kwargs
    ):
        _dtype = torch.float32
        if use_fp16:
            _dtype = torch.float16
        elif use_bf16:
            _dtype = torch.bfloat16
        self.model = SentenceTransformer(model_name_or_path, trust_remote_code=trust_remote_code, model_kwargs={"torch_dtype": _dtype}, **kwargs)
        self.model_card_data = self.model.model_card_data
        self.similarity_fn_name = self.model.similarity_fn_name
        
        self.model = MultiGPUModel(self.model)
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        
        self.prompt_template = prompt_template
        if prompt_template:
            assert (
                "{}" in prompt_template
            ), "Prompt template must contain '{}' placeholder."
        self.prompts_dict = prompts_dict or {}
        self.assert_prompts_exist = assert_prompts_exist
        
        self.only_download_data = only_download_data

    def get_prompt(
        self,
        task_name: str,
    ):
        if task_name in self.prompts_dict:
            instruction = self.prompts_dict[task_name]
        else:
            _prompt_value = mteb.get_task(task_name).metadata.prompt
            if _prompt_value:
                if isinstance(_prompt_value, str):
                    instruction = _prompt_value
                else:
                    instruction = _prompt_value["query"]
            else:
                if self.assert_prompts_exist:
                    raise ValueError(f"Prompt for task '{task_name}' not found in prompts_dict or task metadata.")
                instruction = "Given the following text, generate its embedding."
        
        prompt = (
            self.prompt_template.format(instruction)
            if self.prompt_template
            else instruction
        )
        return prompt

    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type = None, # "query" or "document"
        **kwargs
    ):
        if self.only_download_data:
            raise RuntimeError("Model is set to only download data, not perform encoding.")

        self.model.model.max_seq_length = self.max_length
        
        if prompt_type == "document":
            return self.model.encode(
                sentences,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=True,
            )
        else:
            prompt = self.get_prompt(task_name)
            return self.model.encode(
                sentences,
                prompt=prompt,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=True,
            )
