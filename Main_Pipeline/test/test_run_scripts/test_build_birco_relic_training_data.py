import os
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = PROJECT_ROOT / "run-scripts" / "0802-build-birco-relic-training-data.sh"

PROMPT = (
    "Given a literary analysis excerpt with a masked quotation, retrieve passages "
    "from the analyzed work that fit the gap and support the surrounding "
    "interpretation."
)


def test_dry_run_uses_ordinary_converter_and_correct_inputs() -> None:
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
    assert output.count("anno_queries_to_msmarco_syn.py") == 1
    assert "Annotated_Main_20260801_233302.json" in output
    assert "DiverseQuery_Main_20260801_205707.json" in output
    assert PROMPT in output
    assert "birco-relic-20260801-gen-query.jsonl" in output
    assert (
        "birco-relic-20260801-gen-query_exclude_other_pos_save_non_pos.jsonl"
        in output
    )


if __name__ == "__main__":
    test_dry_run_uses_ordinary_converter_and_correct_inputs()
    print("PASS: BIRCO-RELIC training-data build script")
