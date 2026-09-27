import mteb
from transformers import HfArgumentParser

from arguments import EvalArgs
from task_prompts import PROMPTS_DICT
from sentence_transformers_model import SentenceTransformerEncoder, FlagEmbeddingEncoder


def get_model(args, prompts_dict=None):
    model = SentenceTransformerEncoder(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fp16=args.use_fp16,
        use_bf16=args.use_bf16,
        normalize_embeddings=args.normalize_embeddings,
        max_length=args.max_length,
        batch_size=args.batch_size,
        prompt_template=args.prompt_template,
        prompts_dict=prompts_dict,
        assert_prompts_exist=args.assert_prompts_exist,
        only_download_data=args.only_download_data,
    )
    return model


def main(args):
    # select benchmark
    print("Loading benchmark:", args.benchmark_name)
    # benchmark = mteb.get_benchmark(args.benchmark_name)
    # filter Retrieval Tasks
    tasks = []
    # for task in benchmark.tasks:
    #     if task.metadata.type == "Retrieval":
    #         print("Including task:", task.metadata.name)
    #         tasks.append(task)
            
    if args.benchmark_name.startswith("MTEB(") or args.benchmark_name.startswith("NanoBEIR"):
        # 还是原来的 benchmark 逻辑
        benchmark = mteb.get_benchmark(args.benchmark_name)
        tasks = [
            task for task in benchmark.tasks if task.metadata.type == "Retrieval"
        ]
        print(f"[INFO] Loaded benchmark with {len(tasks)} retrieval tasks")
    else:
        # 其他情况一律当成「单个任务名」
        print(f"[INFO] Interpreting '{args.benchmark_name}' as a single task")
        task = mteb.get_task(args.benchmark_name)
        tasks = [task]
    
    # get model
    print("Loading model:", args.model_name_or_path)
    model = get_model(args, prompts_dict=PROMPTS_DICT)
    
    if args.only_download_data:
        for task in tasks:
            print("Downloading data for task:", task.metadata.name)
            evaluation = mteb.MTEB(tasks=[task])
            try:
                evaluation.run(model)
            except RuntimeError as e:
                pass
        print("Data download completed.")
        return
    
    # run evaluation
    print("Running evaluation...")
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(
        model,
        save_predictions=True,
        output_folder=args.results_output_folder,
    )
    
    print("Evaluation completed. Results saved to:", args.results_output_folder)
    print("All done!")


if __name__ == "__main__":
    parser = HfArgumentParser(EvalArgs)
    
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
