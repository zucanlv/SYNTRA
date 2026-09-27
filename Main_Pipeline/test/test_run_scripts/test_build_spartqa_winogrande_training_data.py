import os
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = (
    PROJECT_ROOT
    / "run-scripts"
    / "0801-build-spartqa-winogrande-training-data.sh"
)

SPARTQA_PROMPT = (
    "Given a scene description and a multiple-choice spatial question, retrieve "
    "the candidate answer option supported by the stated relations and applicable "
    "spatial inference rules, including container-to-content relation inheritance."
)
WINOGRANDE_PROMPT = (
    "Given a sentence with a blank marked by an underscore, retrieve the answer "
    "phrase that yields the intended commonsense-plausible completion when inserted "
    "into the blank."
)


def test_dry_run_uses_ordinary_converter_and_correct_result_pairs() -> None:
    with tempfile.TemporaryDirectory() as dirname:
        output_dir = Path(dirname) / "output"
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_CONDA_ACTIVATE": "1",
                "OUTPUT_DIR": str(output_dir),
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
    assert output.count("anno_queries_to_msmarco_syn.py") == 4
    assert SPARTQA_PROMPT in output
    assert WINOGRANDE_PROMPT in output

    expected_inputs = (
        "Annotated_Main_20260801_050243.json",
        "DiverseQuery_Main_20260731_235706.json",
        "Annotated_Main_20260731_210630.json",
        "DiverseQuery_FromFile_20260731_210619.json",
        "Annotated_Main_20260801_050247.json",
        "DiverseQuery_Main_20260801_000232.json",
        "Annotated_Main_20260801_050258.json",
        "DiverseQuery_FromFile_20260801_045658.json",
    )
    for input_name in expected_inputs:
        assert input_name in output

    expected_outputs = (
        "spartqa-mchoice-20260801-gen-query.jsonl",
        "spartqa-mchoice-20260801-ori-query.jsonl",
        "winogrande-20260801-gen-query.jsonl",
        "winogrande-20260801-ori-query.jsonl",
    )
    for output_name in expected_outputs:
        assert output_name in output
        assert output_name.replace(
            ".jsonl", "_exclude_other_pos_save_non_pos.jsonl"
        ) in output


if __name__ == "__main__":
    test_dry_run_uses_ordinary_converter_and_correct_result_pairs()
    print("PASS: SpartQA/WinoGrande training-data build script")
