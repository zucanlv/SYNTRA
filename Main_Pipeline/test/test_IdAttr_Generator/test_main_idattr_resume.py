import ast
from pathlib import Path


MAIN_PIPELINE_DIR = Path(__file__).resolve().parents[2]
MAIN_SOURCE_PATH = MAIN_PIPELINE_DIR / "main.py"


def _is_args_idattr_resume_path(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "idattr_resume_path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    )


def test_main_wires_idattr_resume_path_from_cli_to_generator_and_run_config():
    tree = ast.parse(MAIN_SOURCE_PATH.read_text(encoding="utf-8"))

    parser_argument_found = False
    generator_wiring_found = False
    run_config_wiring_found = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and any(
                    isinstance(arg, ast.Constant)
                    and arg.value == "--idattr-resume-path"
                    for arg in node.args
                )
            ):
                parser_argument_found = True

            if isinstance(node.func, ast.Name) and node.func.id == "IdAttrConfig":
                generator_wiring_found = any(
                    keyword.arg == "resume_path"
                    and _is_args_idattr_resume_path(keyword.value)
                    for keyword in node.keywords
                )

        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "idattr_resume_path"
                    and _is_args_idattr_resume_path(value)
                ):
                    run_config_wiring_found = True

    assert parser_argument_found
    assert generator_wiring_found
    assert run_config_wiring_found


if __name__ == "__main__":
    test_main_wires_idattr_resume_path_from_cli_to_generator_and_run_config()
