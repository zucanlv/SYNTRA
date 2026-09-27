from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("run-evidencebench.sh")

FILTERED_DATA = (
    "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/"
    "evidencebench/evidencebench-10k-1q-20260806_"
    "exclude_other_pos_save_non_pos.jsonl"
)
RAW_DATA = (
    "/data/share/project/shared_datasets/DSA/syn_data/more-domain-task/"
    "evidencebench/evidencebench-10k-1q-20260806.jsonl"
)


def test_evidencebench_workflow_uses_the_intended_training_data():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert FILTERED_DATA in script
    assert RAW_DATA in script
    assert "birco-relic" not in script.lower()


def test_evidencebench_workflow_trains_then_evaluates_its_merged_model():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    train_call = script.index("train-qwen3_0.6b.sh")
    eval_call = script.index("run_flagembedding_eval.sh")
    assert train_call < eval_call
    assert "train-qwen3_0.6b-evidencebench" in script
    assert 'converted_full/evidencebench' in script
    assert '--query-instruction-format' in script


if __name__ == "__main__":
    test_evidencebench_workflow_uses_the_intended_training_data()
    test_evidencebench_workflow_trains_then_evaluates_its_merged_model()
    print("PASS: EvidenceBench workflow")
