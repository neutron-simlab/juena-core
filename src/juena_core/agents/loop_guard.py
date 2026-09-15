"""Stub for 01/CP2. Ported from ``juena/agents/loop_guard.py``.

**Do not rename ``GUARD_FLAG``** (00-BOUNDARY.md, *Two literals*) — it is a
checkpointed value on live threads. Change the comment explaining why the
name looks wrong; keep the string.
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware

__all__ = ["GUARD_FLAG", "REPEAT_ERROR", "STOPPED", "REPEAT_LIMIT", "RepeatedToolCallMiddleware"]

GUARD_FLAG = "juena_repeated_tool_call"
REPEAT_ERROR = ""
STOPPED = ""
REPEAT_LIMIT = 2


class RepeatedToolCallMiddleware(AgentMiddleware):
    """Stub — implemented in 01/CP2."""
