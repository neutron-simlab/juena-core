"""Request-scoped capture of typed sandbox execution evidence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from juena_core.schema.interrupts import ExecutionEvidence

__all__ = ["capture_sandbox_execution", "record_sandbox_execution"]

_Capture = tuple[str, list[ExecutionEvidence]]
_capture: ContextVar[_Capture | None] = ContextVar(
    # Stored name retained for tracing and compatibility with existing logs.
    "juena_sandbox_execution_capture",
    default=None,
)


@contextmanager
def capture_sandbox_execution(graph_run_id: str) -> Iterator[list[ExecutionEvidence]]:
    """Capture backend evidence produced by one direct execute tool call."""

    events: list[ExecutionEvidence] = []
    token = _capture.set((graph_run_id, events))
    try:
        yield events
    finally:
        _capture.reset(token)


def record_sandbox_execution(**values: Any) -> None:
    """Append evidence only while execution middleware owns a capture."""

    active = _capture.get()
    if active is None:
        return
    graph_run_id, events = active
    events.append(
        ExecutionEvidence.model_validate(
            {"graph_run_id": graph_run_id, **values}
        )
    )
