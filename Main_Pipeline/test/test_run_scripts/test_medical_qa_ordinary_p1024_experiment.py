import os
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_SCRIPT = (
    PROJECT_ROOT
    / "run-scripts"
    / "0816-run-medical-qa-ordinary-passage1024.sh"
)

RAW_OUTPUTS = (
    "medical_qa-all-1q-20260806.jsonl",
    "medical_qa-all-2q-query1-20260811.jsonl",
    "medical_qa-all-2q-query2-20260811.jsonl",
    "medical_qa-all-4q-query1-20260812.jsonl",
    "medical_qa-all-4q-query2-20260812.jsonl",
    "medical_qa-all-4q-query3-20260812.jsonl",
    "medical_qa-all-4q-query4-20260812.jsonl",
)


def _run(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_ordinary_data_experiment_trains_and_evaluates_at_passage_1024() -> None:
    with tempfile.TemporaryDirectory() as dirname:
        temp_root = Path(dirname)
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_CONDA_ACTIVATE": "1",
                "MODEL_ROOT": str(temp_root / "models"),
                "EVAL_ROOT": str(temp_root / "eval"),
                "LOG_ROOT": str(temp_root / "logs"),
            }
        )
        result = _run(EXPERIMENT_SCRIPT, env)

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert output.count("DRY_RUN TRAIN") == 3
    assert output.count("DRY_RUN EVAL") == 3
    assert output.count("--passage_max_len 1024") == 3
    assert output.count("--max_length 1024") == 3
    assert output.count("--query_max_len 512") == 3
    assert output.count("--num_train_epochs 1") == 3
    assert output.count("--train_group_size 8") == 3
    assert output.count("--per_device_train_batch_size 16") == 3
    assert output.count(
        "/data/share/project/public_envs/embedder_eval_latest/bin/python"
    ) == 3
    assert "fill-easy-7" not in output

    positions = [output.index(f"experiment={name}") for name in ("1q", "2q", "4q")]
    assert positions == sorted(positions)
    for output_name in RAW_OUTPUTS:
        assert output_name in output
        assert output_name.replace(
            ".jsonl", "_exclude_other_pos_save_non_pos.jsonl"
        ) in output


def test_resume_skips_existing_ordinary_training_but_still_evaluates() -> None:
    with tempfile.TemporaryDirectory() as dirname:
        temp_root = Path(dirname)
        (temp_root / "models" / "1q" / "merged_model").mkdir(parents=True)
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_CONDA_ACTIVATE": "1",
                "SKIP_EXISTING_TRAIN": "1",
                "MODEL_ROOT": str(temp_root / "models"),
                "EVAL_ROOT": str(temp_root / "eval"),
                "LOG_ROOT": str(temp_root / "logs"),
            }
        )
        result = _run(EXPERIMENT_SCRIPT, env)

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "SKIP TRAIN experiment=1q" in output
    assert "DRY_RUN TRAIN experiment=1q" not in output
    assert "DRY_RUN EVAL experiment=1q" in output
    assert "DRY_RUN TRAIN experiment=2q" in output
    assert "DRY_RUN TRAIN experiment=4q" in output


if __name__ == "__main__":
    test_ordinary_data_experiment_trains_and_evaluates_at_passage_1024()
    test_resume_skips_existing_ordinary_training_but_still_evaluates()
    print("PASS: MedicalQA ordinary-data passage-1024 experiment")
