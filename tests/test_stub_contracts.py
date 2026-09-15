"""Structural contracts fixed by the CP0 package skeleton."""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path

from juena_core.clients.base import BaseAgentClient
from juena_core.server.agent.registry import DEFAULT_AGENT
from juena_core.ui.chat_storage import Chat

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "juena_core"


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


def test_every_declared_module_export_has_a_stub() -> None:
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        definitions: set[str] = set()
        exports: list[str] | None = None

        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                definitions.update(_assigned_names(node))
                if "__all__" in _assigned_names(node):
                    value = node.value
                    exports = ast.literal_eval(value) if value is not None else None
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                definitions.update(alias.asname or alias.name.split(".")[0] for alias in node.names)

        assert exports is not None, f"{path.relative_to(PACKAGE_ROOT)} has no __all__"
        for name in exports:
            child_module = path.parent / f"{name}.py"
            child_package = path.parent / name / "__init__.py"
            assert name in definitions or child_module.is_file() or child_package.is_file(), (
                f"{path.relative_to(PACKAGE_ROOT)} exports undefined name {name!r}"
            )


def test_base_agent_client_keeps_the_synchronous_ui_contract() -> None:
    methods = (
        "health",
        "list_chats",
        "create_chat",
        "get_chat",
        "update_chat",
        "get_artifact",
        "get_pending_interrupt",
        "delete_thread",
    )
    assert all(
        not inspect.iscoroutinefunction(getattr(BaseAgentClient, name)) for name in methods
    )
    assert inspect.isgeneratorfunction(BaseAgentClient.stream)
    assert inspect.isgeneratorfunction(BaseAgentClient.resume_stream)


def test_ui_chat_records_its_agent_identity() -> None:
    assert "agent_id" in {field.name for field in fields(Chat)}


def test_shared_registry_has_no_application_specific_default() -> None:
    assert DEFAULT_AGENT is None
