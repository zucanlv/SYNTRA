"""Read-only checks for the original-workflow delivery configuration.

Run from Main_Pipeline. No generation, network requests, model downloads or
directory creation. This checks the documented full annotation route only;
historical resume/ablation recipes have different prerequisites.
"""
import argparse
import importlib
import os
from pathlib import Path
import sys

PIPELINE = Path(__file__).resolve().parents[1] / "Main_Pipeline"


def check_config(cfg, phase, env, pipeline=PIPELINE):
    errors = []
    if not isinstance(cfg, dict):
        return ["Configuration must be a YAML mapping."]
    sys.path.insert(0, str(pipeline))
    from HighLevel_Def import HighLevel_Task_Definition
    from Few_Shot_Example import Few_Shot_Example

    task = cfg.get("task")
    if task not in HighLevel_Task_Definition or task not in Few_Shot_Example:
        errors.append("task must exist in both HighLevel_Def.py and Few_Shot_Example.py.")
    # Never interpolate secrets or configuration values into error messages.
    def visit(value, key="config"):
        if isinstance(value, dict):
            for name, item in value.items():
                if "api_key" in str(name) and item:
                    errors.append("Keep API keys out of YAML; use OPENAI_API_KEY.")
                visit(item, str(name))
        elif isinstance(value, list):
            for item in value:
                visit(item, key)
        elif isinstance(value, str):
            if "REPLACE_" in value or "REDACTED_" in value:
                errors.append(f"Replace the placeholder in {key}.")
            if "/data/share/project/" in value or "/home/ustc/" in value:
                errors.append(f"Replace the lab-specific path in {key}.")
            if "${" in value:
                errors.append(f"YAML does not expand environment variables in {key}.")
    visit(cfg)
    corpus = Path(str(cfg.get("corpus_path") or ""))
    if not corpus.is_absolute() or not corpus.exists():
        errors.append("corpus_path must be an existing absolute path.")
    embedding = cfg.get("embedding_model_path")
    if embedding and not Path(str(embedding)).is_dir():
        errors.append("embedding_model_path must be an existing model directory, or null.")
    for field in ("faiss_dir", "instructions_store_dir"):
        value = cfg.get(field)
        if not value or not Path(str(value)).is_absolute():
            errors.append(f"Set {field} to a dedicated absolute output path.")
        elif Path(value).resolve() == corpus.resolve():
            errors.append(f"{field} must not be the corpus path.")
    stages = cfg.get("llm_stages") or {}
    if not isinstance(stages, dict):
        errors.append("llm_stages must be a mapping.")
        stages = {}
    for stage in ("idattr_refine", "idattr_gen", "queryinstr", "diversequery", "annoinstr", "annotation"):
        values = stages.get(stage) or {}
        if not isinstance(values, dict):
            errors.append(f"llm_stages.{stage} must be a mapping.")
            continue
        if not (values.get("model") or cfg.get("llm_model")):
            errors.append(f"Set a model for {stage}.")
        if not (values.get("model_server") or cfg.get("llm_model_server") or env.get("OPENAI_BASE_URL")):
            errors.append(f"Set an endpoint for {stage}, or OPENAI_BASE_URL.")
    if not env.get("OPENAI_API_KEY"):
        errors.append("Set OPENAI_API_KEY (a local endpoint may accept a dummy value).")
    for field in ("step4_diverse_query_input", "step5_query2passage_input", "diverse_texts", "filtered_diverse_texts"):
        if cfg.get(field):
            errors.append(f"The documented from-corpus route requires {field}: null.")
    if cfg.get("skip_index") or cfg.get("fully_annotation") is not True:
        errors.append("This delivery route requires skip_index: false and fully_annotation: true.")
    if phase == "production":
        if cfg.get("test_mode"):
            errors.append("Set test_mode: false before production; --test-mode cannot disable it.")
        store = Path(str(cfg.get("instructions_store_dir") or ""))
        for prefix in ("IdAttrRefine", "QueryInstrRefine", "AnnoInstrRefine"):
            candidates = [p for p in store.glob(f"{prefix}_{task}_*.txt")
                          if "_attempt" not in p.name and "_history" not in p.name]
            if not any(p.is_file() and p.read_text().strip() for p in candidates):
                errors.append(f"No nonempty final {prefix} instruction; run and inspect refinement first.")
    return list(dict.fromkeys(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("refine", "production"), required=True)
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != PIPELINE.resolve():
        parser.exit(1, "Run this check and main.py from Main_Pipeline.\n")
    try:
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
    except ImportError:
        parser.exit(1, "Install PyYAML from requirements-synthesis.txt.\n")
    except (OSError, yaml.YAMLError):
        parser.exit(1, "Cannot read valid YAML from --config; check the file and syntax.\n")
    errors = check_config(cfg, args.phase, os.environ)
    if args.check_imports:
        sys.path.insert(0, str(PIPELINE))
        try:
            importlib.import_module("main")
        except Exception as exc:
            errors.append(f"Pipeline import failed ({type(exc).__name__}); verify the documented environment.")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print("PASS: local prerequisites checked. No API/model/data generation was run.")
    print("Inspect the corpus size: --test-mode still indexes the supplied corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
