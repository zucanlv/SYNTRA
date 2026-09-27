import os
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = (
    PROJECT_ROOT
    / "run-scripts"
    / "0816-build-theoremqa-arguana-training-data.sh"
)

THEOREMQA_PROMPT = (
    "Given a Math problem, retrieve relevant theorems that help answer the problem."
)
ARGUANA_PROMPT = "Given a claim, find documents that refute the claim."


def test_dry_run_builds_three_ordinary_training_data_pairs() -> None:
    with tempfile.TemporaryDirectory() as dirname:
        temp_root = Path(dirname)
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_CONDA_ACTIVATE": "1",
                "THEOREMQA_OUTPUT_DIR": str(temp_root / "theoremqa"),
                "ARGUANA_OUTPUT_DIR": str(temp_root / "arguana"),
            }
        )
        result = subprocess.run(
            ["bash", str(BUILD_SCRIPT)],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "anno_queries_to_msmarco_syn_fill_easy.py" not in output
    assert output.count("anno_queries_to_msmarco_syn.py") == 3
    assert THEOREMQA_PROMPT in output
    assert output.count(ARGUANA_PROMPT) == 2

    expected_inputs = (
        "Annotated_Main_20260815_182128.json",
        "DiverseQuery_Main_20260814_231554.json",
        "Annotated_Main_20260816_015524.json",
        "DiverseQuery_Main_20260815_011304.json",
        "Annotated_Main_20260816_092808.json",
        "DiverseQuery_Main_20260815_030955.json",
    )
    for input_name in expected_inputs:
        assert input_name in output

    expected_outputs = (
        "bright-documents-theoremqa_theorems-10k-20260816-wo-self-refine.jsonl",
        "arguana-20260816-wo-self-refine.jsonl",
        "arguana-20260816-with-self-refine.jsonl",
    )
    for output_name in expected_outputs:
        assert output_name in output
        assert output_name.replace(
            ".jsonl", "_exclude_other_pos_save_non_pos.jsonl"
        ) in output


if __name__ == "__main__":
    test_dry_run_builds_three_ordinary_training_data_pairs()
    print("PASS: TheoremQA/ArguAna ordinary training-data build script")
