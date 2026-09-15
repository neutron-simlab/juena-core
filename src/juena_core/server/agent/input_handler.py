"""Stub for 01/CP4. Ported from ``juena/server/agent/input_handler.py``,
including ``raise ValueError("A trusted authenticated user_id is required")``
— the identity requirement is already enforced here and stays
(00-BOUNDARY.md, *Moves whole*)."""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["AgentRunContext", "AgentInputHandler"]


class AgentRunContext(NamedTuple):
    """Stub — implemented in 01/CP4."""


class AgentInputHandler:
    """Stub — implemented in 01/CP4."""
