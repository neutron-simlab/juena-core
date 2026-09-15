"""The SSE streaming vocabulary: wire payloads, stream-mode handlers, and the
processor that turns one into the other. See 00-BOUNDARY.md, *Moves whole*.

Deliberately re-exports nothing, like ``server/database/__init__.py``. The
processor asks :mod:`juena_core.server.interrupts` how to render a pause, and
that module builds the clarification payload with :mod:`.events` — so a package
``__init__`` that eagerly imported ``processor`` would close the loop through
itself. Import the submodule you need.
"""

from __future__ import annotations

__all__ = ["events", "handlers", "processor"]
