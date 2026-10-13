"""The import-direction rule as a test, not just a grep.

``scripts/check-imports.sh`` is the fast gate for a commit hook; this is the
one that cannot be fooled. It walks every module with ``ast.parse``, so it
catches an import split across lines by parentheses and does not mistake a
matching string inside a docstring for a violation — both of which a
regex can get wrong in either direction.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_ROOTS = {"juena", "vitess_ai", "juena_rag"}
# Tracing vendors belong to the application too. Core only offers the hook in
# ``juena_core.tracing``; which tracer runs is the application's choice.
FORBIDDEN_VENDORS = {"langfuse", "langsmith"}


def _python_files() -> list[Path]:
    return sorted((ROOT / "src").rglob("*.py")) + sorted(
        p for p in (ROOT / "tests").rglob("*.py") if p.name != Path(__file__).name
    )


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_module_does_not_import_an_application(path: Path):
    forbidden = _imported_roots(path) & (FORBIDDEN_ROOTS | FORBIDDEN_VENDORS)
    assert not forbidden, (
        f"{path.relative_to(ROOT)} imports forbidden application or vendor "
        f"module(s): {forbidden}"
    )


def test_core_does_not_declare_a_tracing_vendor() -> None:
    """An unused dependency is the step before an import, so catch it there."""

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = [
        *project["dependencies"],
        *(
            requirement
            for extra in project.get("optional-dependencies", {}).values()
            for requirement in extra
        ),
    ]
    # The distribution name is everything before extras, a version bound or
    # an environment marker: "langfuse>=4" and "langfuse[x]" are both langfuse.
    names = {re.split(r"[\s\[<>=!~;]", item, maxsplit=1)[0] for item in declared}
    assert not names & FORBIDDEN_VENDORS


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import juena\n", {"juena"}),
        ("from vitess_ai.tools import (\n    run,\n)\n", {"vitess_ai"}),
        ("import juena_core\n", {"juena_core"}),
        ('"""import juena_rag"""\n', set()),
    ],
)
def test_import_parser_uses_python_syntax(
    tmp_path: Path,
    source: str,
    expected: set[str],
) -> None:
    path = tmp_path / "sample.py"
    path.write_text(source)
    assert _imported_roots(path) == expected
