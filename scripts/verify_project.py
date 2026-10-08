"""Audit Python syntax, imports, dependency constraints, and academic docstrings.

This intentionally skips executing frontend/app.py; Streamlit AppTest verifies it.
Run from any directory with the project's Python environment.
"""

import ast
import importlib
import importlib.metadata
import sys
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_files() -> list[Path]:
    """Return project Python files, excluding virtual environments and generated caches."""
    paths = []
    for directory in ("backend", "frontend", "tests", "scripts"):
        paths.extend((ROOT / directory).rglob("*.py"))
    return sorted(paths + [ROOT / "evaluation.py"])


def main() -> None:
    """Fail on missing documentation, syntax/import errors, or dependency mismatches."""
    failures = []
    files = source_files()
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
        if not ast.get_docstring(tree):
            failures.append(f"{path.relative_to(ROOT)}: missing module docstring")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    failures.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: missing {node.name} docstring"
                    )
        relative = path.relative_to(ROOT)
        if (
            relative.parts[0] in ("backend", "frontend")
            and relative.as_posix() != "frontend/app.py"
        ):
            module = ".".join(relative.with_suffix("").parts)
            importlib.import_module(module)
    checked = 0
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        installed = importlib.metadata.version(requirement.name)
        if installed not in requirement.specifier:
            failures.append(
                f"{requirement.name}: {installed} does not meet {requirement.specifier}"
            )
        checked += 1
    for name in (
        "CAPSTONE_REPORT.md",
        "VIVA_QUESTIONS.md",
        "PRESENTATION_OUTLINE.md",
        "DEMO_SCRIPT.md",
    ):
        if not (ROOT / "docs" / name).is_file():
            failures.append(f"Missing academic document: {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified syntax and docstrings in {len(files)} Python files.")
    print(
        f"Imported all backend/frontend library modules; checked {checked} dependency constraints."
    )


if __name__ == "__main__":
    main()
