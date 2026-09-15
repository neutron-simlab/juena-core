"""Stub for 01/CP4. Ported from ``juena/server/streaming/processor.py``,
minus one sandbox import (00-BOUNDARY.md, *Moves whole*).

The approval branch that used to check
``interrupt_kind(...) == EXECUTE_APPROVAL_KIND`` becomes a lookup into the
``register_interrupt_event`` dictionary declared in ``server/service.py``
(00-BOUNDARY.md, decision 5) — core knows the clarification kind; the
application registers its own.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Interrupt

__all__ = ["interrupt_event", "StreamEventProcessor"]


def interrupt_event(interrupt: Interrupt) -> dict[str, Any] | None:
    raise NotImplementedError("juena_core.server.streaming.processor.interrupt_event lands in 01/CP4")


def _tool_summary(message: Any) -> str:
    raise NotImplementedError("juena_core.server.streaming.processor._tool_summary lands in 01/CP4")


class StreamEventProcessor:
    """Stub — implemented in 01/CP4."""
