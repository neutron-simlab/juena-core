"""Stub for 01/CP2. Ported from ``juena/agents/specialist_runtime.py``, minus
two sandbox imports (00-BOUNDARY.md, decision 5), plus a new
``build_supervisor_middleware`` (00-BOUNDARY.md, decision 3).

**Edit 1 of four for this checkpoint.** ``build_specialist_middleware`` gains
``execution_middleware: Sequence = ()`` and ``interrupt_on: dict | None =
None``, replacing the source's ``enable_execution_approval: bool`` flag.
juena-chatbot passes its two; v2 passes nothing. **Keep the ``unattended``
coupling intact** — ``ask_user_bound=not unattended`` and the absent
execution backend must keep moving together, or a specialist can run
commands with nobody to approve them.

``SPECIALIST_TASK_DESCRIPTION`` is ``SUPERVISOR_TASK_DESCRIPTION`` from
``juena/agents/juena_agent.py`` (which stays in the application), renamed and
moved here as ``build_supervisor_middleware``'s default ``task_description``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "SUPERVISOR_MODEL_CALL_LIMIT",
    "SUPERVISOR_TOOL_CALL_LIMIT",
    "SPECIALIST_MODEL_CALL_LIMIT",
    "SPECIALIST_TOOL_CALL_LIMIT",
    "BACKGROUND_MODEL_CALL_LIMIT",
    "BACKGROUND_TOOL_CALL_LIMIT",
    "ASK_USER_CALL_LIMIT",
    "MODEL_MAX_RETRIES",
    "TOOL_MAX_RETRIES",
    "UNATTENDED_NOTICE",
    "SPECIALIST_TASK_DESCRIPTION",
    "PromptResourceError",
    "load_markdown",
    "has_authored_skills",
    "build_specialist_backend",
    "resilience_middleware",
    "build_fallback_models",
    "build_specialist_middleware",
    "build_supervisor_middleware",
]

SUPERVISOR_MODEL_CALL_LIMIT = 50
SUPERVISOR_TOOL_CALL_LIMIT = 100
SPECIALIST_MODEL_CALL_LIMIT = 60
SPECIALIST_TOOL_CALL_LIMIT = 150
BACKGROUND_MODEL_CALL_LIMIT = 150
BACKGROUND_TOOL_CALL_LIMIT = 400
ASK_USER_CALL_LIMIT = 3
MODEL_MAX_RETRIES = 3
TOOL_MAX_RETRIES = 2
UNATTENDED_NOTICE = ""
SPECIALIST_TASK_DESCRIPTION = ""


class PromptResourceError(RuntimeError):
    """Stub — implemented in 01/CP2."""


def load_markdown(package: str, filename: str) -> str:
    raise NotImplementedError("juena_core.agents.specialist_runtime.load_markdown lands in 01/CP2")


def has_authored_skills(skills_dir: Path | None) -> bool:
    raise NotImplementedError("juena_core.agents.specialist_runtime.has_authored_skills lands in 01/CP2")


def build_specialist_backend(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.agents.specialist_runtime.build_specialist_backend lands in 01/CP2")


def resilience_middleware(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.agents.specialist_runtime.resilience_middleware lands in 01/CP2")


def build_fallback_models() -> list[Any]:
    raise NotImplementedError("juena_core.agents.specialist_runtime.build_fallback_models lands in 01/CP2")


def build_specialist_middleware(
    *args: Any,
    execution_middleware: Sequence[Any] = (),
    interrupt_on: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Any]:
    raise NotImplementedError("juena_core.agents.specialist_runtime.build_specialist_middleware lands in 01/CP2")


def build_supervisor_middleware(
    *,
    backend: Any,
    summarizer_model: Any,
    fallback_models: Any,
    subagents: Any,
    task_description: str = SPECIALIST_TASK_DESCRIPTION,
    extra: Sequence[Any] = (),
    model_call_limit: int = SUPERVISOR_MODEL_CALL_LIMIT,
    tool_call_limit: int = SUPERVISOR_TOOL_CALL_LIMIT,
) -> list[Any]:
    """New in core (00-BOUNDARY.md, decision 3). Returns the canonical
    middleware stack with exactly one named insertion point: ``extra`` is
    spliced after ``PatchToolCallsMiddleware`` and before
    ``RuntimeModelMiddleware``. Not a hook system — a list, spliced at one
    documented index, everything else fixed. 01/CP2's done-when condition is
    a permanent test asserting the returned list's class names in order.
    """
    raise NotImplementedError("juena_core.agents.specialist_runtime.build_supervisor_middleware lands in 01/CP2")
