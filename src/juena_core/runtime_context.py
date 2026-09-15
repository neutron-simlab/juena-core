"""Stub for 01/CP2. Lifted from ``juena/sandbox/runtime.py::_context_value``
(00-BOUNDARY.md, decision 5) — 5 lines, generic: reads a named attribute off
a runtime context that may be a dict or a dataclass.

**Not** the API for a façade tool's thread identity (03/CP3a). CP0b measured
that ``request.runtime.context`` is ``None`` unless a ``context_schema`` and
invocation ``context`` are supplied, and that public thread identity lives at
``request.runtime.execution_info.thread_id`` instead (00-BOUNDARY.md,
decision 16). This helper remains useful for reading *typed application*
context values (e.g. a principal id) once ``context_schema``/``context=`` are
wired up — it must not be reused to reach for thread identity.
"""

from __future__ import annotations

from typing import Any

__all__ = ["_context_value"]


def _context_value(context: Any, name: str) -> str | None:
    raise NotImplementedError("juena_core.runtime_context._context_value lands in 01/CP2")
