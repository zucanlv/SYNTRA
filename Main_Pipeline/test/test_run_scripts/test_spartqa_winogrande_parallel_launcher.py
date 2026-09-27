import os
import subprocess
import tempfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = PROJECT_ROOT / "run-scripts" / "0731-run-spartqa-winogrande-4-parallel.sh"
CONFIG_DIR = (
    PROJECT_ROOT
    / "Main_Pipeline"
    / "config"
    / "RQ-2-generalization"
    / "reasoning-intensive"
)
CONFIG_NAMES = (
    "config_0731-spartqa-mchoice-gen-query.yaml",
    "config_0731-spartqa-mchoice-train-qpos.yaml",
    "config_0731-winogrande-gen-query.yaml",
    "config_0731-winogrande-train-qpos.yaml",
)


def _make_fake_faiss_dir(root: Path, name: str) -> Path:
    faiss_dir = root / name
    faiss_dir.mkdir()
    for filename in ("index.faiss", "id_map.pkl", "doc_dict.pkl"):
        (faiss_dir / filename).touch()
    return faiss_dir


def _run_dry_run(tmpdir: Path) -> subprocess.CompletedProcess[str]:
    spartqa_faiss_dir = _make_fake_faiss_dir(tmpdir, "spartqa-faiss")
    winogrande_faiss_dir = _make_fake_faiss_dir(tmpdir, "winogrande-faiss")
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "SKIP_CONDA_ACTIVATE": "1",
            "SKIP_LLM_HEALTHCHECK": "1",
            "SPARTQA_FAISS_DIR": str(spartqa_faiss_dir),
            "WINOGRANDE_FAISS_DIR": str(winogrande_faiss_dir),
            "LOG_ROOT": str(tmpdir / "logs"),
            "Q2P_LOCK_PATH": str(tmpdir / "shared-q2p.lock"),
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_builds_four_commands_with_one_q2p_lock() -> None:
    with tempfile.TemporaryDirectory() as dirname:
        result = _run_dry_run(Path(dirname))

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    for config_name in CONFIG_NAMES:
        assert config_name in output
    assert output.count("--skip-index") == 4
    assert output.count("--q2p-lock-path") == 4
    assert output.count("shared-q2p.lock") == 5
    assert "DRY_RUN complete: validated 4 commands" in output



def test_default_start_stagger_is_thirty_seconds() -> None:
    with tempfile.TemporaryDirectory() as dirname:
        result = _run_dry_run(Path(dirname))

    assert result.returncode == 0, result.stderr
    assert "Start stagger: 30s" in result.stdout


def test_all_four_configs_use_64_workers() -> None:
    for config_name in CONFIG_NAMES:
        with (CONFIG_DIR / config_name).open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        assert config["concurrency"]["max_workers"] == 64, config_name


def test_preflight_rejects_incomplete_faiss_index() -> None:
    with tempfile.TemporaryDirectory() as dirname:
        tmpdir = Path(dirname)
        spartqa_faiss_dir = _make_fake_faiss_dir(tmpdir, "spartqa-faiss")
        winogrande_faiss_dir = _make_fake_faiss_dir(tmpdir, "winogrande-faiss")
        (winogrande_faiss_dir / "index.faiss").unlink()
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "SKIP_CONDA_ACTIVATE": "1",
                "SKIP_LLM_HEALTHCHECK": "1",
                "SPARTQA_FAISS_DIR": str(spartqa_faiss_dir),
                "WINOGRANDE_FAISS_DIR": str(winogrande_faiss_dir),
                "LOG_ROOT": str(tmpdir / "logs"),
                "Q2P_LOCK_PATH": str(tmpdir / "shared-q2p.lock"),
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

    assert result.returncode != 0
    assert "Missing FAISS artifact" in result.stderr


if __name__ == "__main__":
    test_dry_run_builds_four_commands_with_one_q2p_lock()
    test_default_start_stagger_is_thirty_seconds()
    test_all_four_configs_use_64_workers()
    test_preflight_rejects_incomplete_faiss_index()
    print("PASS: spartqa/winogrande parallel launcher")
