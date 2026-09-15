"""Stub for 01/CP4. Ported from ``juena/server/streaming/handlers.py``
(00-BOUNDARY.md, *Moves whole*)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.types import Interrupt

__all__ = ["is_subagent_namespace", "extract_update_messages", "extract_interrupts", "token_text"]


def is_subagent_namespace(node_path: Any) -> bool:
    raise NotImplementedError("juena_core.server.streaming.handlers.is_subagent_namespace lands in 01/CP4")


def extract_update_messages(event: dict[str, Any]) -> list[BaseMessage]:
    raise NotImplementedError("juena_core.server.streaming.handlers.extract_update_messages lands in 01/CP4")


def extract_interrupts(event: dict[str, Any]) -> list[Interrupt]:
    raise NotImplementedError("juena_core.server.streaming.handlers.extract_interrupts lands in 01/CP4")


def token_text(event: tuple[BaseMessage, dict[str, Any]]) -> str | None:
    raise NotImplementedError("juena_core.server.streaming.handlers.token_text lands in 01/CP4")
