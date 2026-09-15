"""Stub for 01/CP1. Ported from ``juena/schema/server.py`` — all 138 lines
read; nothing names JüNA or a domain (00-BOUNDARY.md, decision 2).

**Not unchanged:** ``CreateChatInput`` gains a required ``agent_id``
(00-BOUNDARY.md, decision 13; 01/CP3) — v2 has two agents, and without it
``/{agent_id}/stream`` would resume any thread under any graph.
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel

__all__ = [
    "UserInput",
    "StreamInput",
    "ToolCall",
    "ChatMessage",
    "HealthStatus",
    "CreateChatInput",
    "UpdateChatInput",
]


class UserInput(BaseModel):
    """Stub — implemented in 01/CP1."""


class StreamInput(UserInput):
    """Stub — implemented in 01/CP1."""


class ToolCall(TypedDict):
    """Stub — implemented in 01/CP1."""


class ChatMessage(BaseModel):
    """Stub — implemented in 01/CP1. The shared transcript type on both ends
    of the wire; the UI parses server-sent events back into it."""


class HealthStatus(BaseModel):
    """Stub — implemented in 01/CP1."""


class CreateChatInput(BaseModel):
    """Stub — implemented in 01/CP3, with a required ``agent_id`` field the
    source model does not have (00-BOUNDARY.md, decision 13)."""


class UpdateChatInput(BaseModel):
    """Stub — implemented in 01/CP1."""
