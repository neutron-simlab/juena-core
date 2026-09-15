"""Stub for 01/CP4. Ported from ``juena/server/streaming/events.py``
(00-BOUNDARY.md, *Moves whole*)."""

from __future__ import annotations

from typing import Any

from juena_core.schema.server import ChatMessage

__all__ = [
    "PHASE_THINKING",
    "PHASE_RESEARCHING",
    "PHASE_TOOL",
    "PHASE_DONE",
    "thread_event",
    "token_event",
    "message_event",
    "status_event",
    "error_event",
    "clarification_required_event",
    "approval_required_event",
]

PHASE_THINKING = "thinking"
PHASE_RESEARCHING = "researching"
PHASE_TOOL = "tool"
PHASE_DONE = "done"


def thread_event(thread_id: str) -> dict[str, Any]:
    raise NotImplementedError("juena_core.server.streaming.events.thread_event lands in 01/CP4")


def token_event(content: str) -> dict[str, Any]:
    raise NotImplementedError("juena_core.server.streaming.events.token_event lands in 01/CP4")


def message_event(message: ChatMessage) -> dict[str, Any]:
    raise NotImplementedError("juena_core.server.streaming.events.message_event lands in 01/CP4")


def status_event(phase: str, label: str, **extra: Any) -> dict[str, Any]:
    raise NotImplementedError("juena_core.server.streaming.events.status_event lands in 01/CP4")


def error_event(content: str) -> dict[str, Any]:
    raise NotImplementedError("juena_core.server.streaming.events.error_event lands in 01/CP4")


def clarification_required_event(interrupt_id: str, value: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError("juena_core.server.streaming.events.clarification_required_event lands in 01/CP4")


def approval_required_event(interrupt_id: str, value: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError("juena_core.server.streaming.events.approval_required_event lands in 01/CP4")
