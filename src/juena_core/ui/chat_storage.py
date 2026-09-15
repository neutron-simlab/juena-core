"""Stub for 01/CP5. Ported from ``juena-chatbot/app/chat_storage.py`` — a
thin synchronous adapter over ``/chats``, which core owns (00-BOUNDARY.md,
decision 7).

Typed against ``BaseAgentClient``, not the source's ``AgentClient`` — core
does not have the application's subclass. ``Chat`` includes ``agent_id`` so
the UI can restore the correct graph rather than guess from page state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from juena_core.clients.base import BaseAgentClient

__all__ = ["Chat", "ChatStorage", "get_chat_storage"]


@dataclass
class Chat:
    """Stub — implemented in 01/CP5."""

    thread_id: str
    agent_id: str
    title: str = "New Chat"
    summary: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


def _parse_datetime(value: Any, fallback: datetime | None = None) -> datetime:
    raise NotImplementedError("juena_core.ui.chat_storage._parse_datetime lands in 01/CP5")


def _chat_from_payload(payload: dict[str, Any]) -> Chat:
    raise NotImplementedError("juena_core.ui.chat_storage._chat_from_payload lands in 01/CP5")


class ChatStorage:
    """Stub — implemented in 01/CP5."""


def get_chat_storage(client: BaseAgentClient) -> ChatStorage:
    raise NotImplementedError("juena_core.ui.chat_storage.get_chat_storage lands in 01/CP5")
