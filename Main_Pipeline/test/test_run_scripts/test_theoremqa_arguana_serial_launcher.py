import os
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = (
    PROJECT_ROOT
    / "run-scripts"
    / "0814-run-theoremqa-arguana-3-serial.sh"
)
CONFIG_NAMES = (
    "config_0814-bright-documents-theoremqa_theorems-10k-wo-self-refine.yaml",
    "config_0811-arguana-wo-self-refine.yaml",
    "config_0811-arguana-all.yaml",
)


def test_dry_run_builds_three_pipeline_commands_in_order() -> None:
    env = os.environ.copy()
    env.update({"DRY_RUN": "1", "SKIP_CONDA_ACTIVATE": "1"})
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    command_lines = [
        line for line in output.splitlines() if line.startswith("DRY_RUN:")
    ]
    assert len(command_lines) == 3
    positions = [output.index(config_name) for config_name in CONFIG_NAMES]
    assert positions == sorted(positions)
    for config_name, command_line in zip(CONFIG_NAMES, command_lines):
        assert "Main_Pipeline/main.py" in command_line
        assert "--config" in command_line
        assert config_name in command_line
        assert not command_line.rstrip().endswith("&")
    assert "DRY_RUN complete: validated 3 serial runs" in output


def test_failed_experiment_does_not_block_later_experiments() -> None:
    with tempfile.TemporaryDirectory() as dirname:
        tmpdir = Path(dirname)
        run_log = tmpdir / "runs.log"
        fake_python = tmpdir / "fake-python"
        fake_python.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
config_path="${@: -1}"
basename "${config_path}" >> "${FAKE_RUN_LOG}"
if [[ "${config_path}" == *theoremqa_theorems* ]]; then
  exit 7
fi
""",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = os.environ.copy()
        env.update(
            {
                "SKIP_CONDA_ACTIVATE": "1",
                "PYTHON_BIN": str(fake_python),
                "FAKE_RUN_LOG": str(run_log),
            }
        )
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        executed_configs = run_log.read_text(encoding="utf-8").splitlines()

    assert result.returncode == 1
    assert executed_configs == list(CONFIG_NAMES)
    output = result.stdout + result.stderr
    assert "Failed 1/3" in output
    assert "Completed 2/3" in output
    assert "Completed 3/3" in output
    assert "1 failed" in output


if __name__ == "__main__":
    test_dry_run_builds_three_pipeline_commands_in_order()
    test_failed_experiment_does_not_block_later_experiments()
    print("PASS: theoremqa/arguana serial launcher")
