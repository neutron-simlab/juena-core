"""Stub for 01/CP2. Ported from ``juena/agents/backends.py``
(00-BOUNDARY.md, *Moves whole*).

``JUENA_MEMORY_SYSTEM_PROMPT`` is renamed ``MEMORY_SYSTEM_PROMPT`` and becomes
an argument to ``build_supervisor_backend`` with the current text as its
default — it is not JüNA-specific prose, it is a prompt-injection guard.
"""

from __future__ import annotations

from typing import Any

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import StateBackend

__all__ = [
    "MEMORY_NAMESPACE_PREFIX",
    "MEMORY_SOURCES",
    "FINDINGS_PREFIX",
    "MEMORY_SYSTEM_PROMPT",
    "SUPERVISOR_FILESYSTEM_TOOL_DESCRIPTIONS",
    "ReadOnlyFilesystemBackend",
    "ReadOnlyInputsStateBackend",
    "FindingsStateBackend",
    "ReadOnlyFindingsStateBackend",
    "SupervisorStateBackend",
    "user_store_namespace",
    "build_supervisor_backend",
]

MEMORY_NAMESPACE_PREFIX = "memories"
MEMORY_SOURCES = ["/memories/AGENTS.md"]
FINDINGS_PREFIX = "/findings/"
MEMORY_SYSTEM_PROMPT = ""
SUPERVISOR_FILESYSTEM_TOOL_DESCRIPTIONS: dict[str, str] = {}


class ReadOnlyFilesystemBackend(FilesystemBackend):
    """Stub — implemented in 01/CP2."""


class ReadOnlyInputsStateBackend(StateBackend):
    """Stub — implemented in 01/CP2."""


class FindingsStateBackend(StateBackend):
    """Stub — implemented in 01/CP2."""


class ReadOnlyFindingsStateBackend(FindingsStateBackend):
    """Stub — implemented in 01/CP2."""


class SupervisorStateBackend(StateBackend):
    """Stub — implemented in 01/CP2."""


def user_store_namespace(runtime: Any) -> tuple[str, str]:
    raise NotImplementedError("juena_core.agents.backends.user_store_namespace lands in 01/CP2")


def build_supervisor_backend(store: Any, *, memory_system_prompt: str = MEMORY_SYSTEM_PROMPT) -> CompositeBackend:
    raise NotImplementedError("juena_core.agents.backends.build_supervisor_backend lands in 01/CP2")
