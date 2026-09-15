"""Authenticated chat-history adapter for the Streamlit UI.

Ported from ``juena-chatbot/app/chat_storage.py`` (00-BOUNDARY.md, decision 7).
Postgres persistence is owned by the FastAPI service; Streamlit keeps no local
conversation database and uses this small synchronous adapter instead.

Two changes from the source, both consequences of CP3's ``chats.agent_id``:

- :class:`Chat` carries ``agent_id``, so restoring a conversation restores the
  graph it belongs to rather than whatever the page happens to have selected.
  Opening a thread under the wrong agent answers 404 on every later request.
- :meth:`ChatStorage.list_chats` takes an optional agent filter, so a UI that
  serves several agents can show one mode's conversations at a time.

Typed against :class:`~juena_core.clients.base.BaseAgentClient`, not the
source's ``AgentClient``: core does not have an application's subclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from juena_core.clients.base import BaseAgentClient
from juena_core.schema.server import ChatMessage

__all__ = ["Chat", "ChatStorage", "get_chat_storage"]


@dataclass
class Chat:
    """One user-owned conversation, as the sidebar knows it."""

    thread_id: str
    agent_id: str
    title: str = "New Chat"
    summary: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


def _parse_datetime(value: Any, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return fallback or datetime.now()


def _chat_from_payload(payload: dict[str, Any]) -> Chat:
    return Chat(
        thread_id=str(payload["thread_id"]),
        agent_id=str(payload["agent_id"]),
        title=str(payload.get("title") or "New Chat"),
        summary=str(payload.get("summary") or ""),
        created_at=_parse_datetime(payload.get("created_at")),
        updated_at=_parse_datetime(payload.get("updated_at")),
    )


class ChatStorage:
    """Remote storage facade backed by the authenticated chat endpoints."""

    def __init__(self, client: BaseAgentClient) -> None:
        self.client = client

    def upsert_chat(self, chat: Chat) -> None:
        # Metadata-only probe: the caller is deciding create-vs-update and has
        # no use for the thread's message history.
        existing = self.client.get_chat(chat.thread_id, include_messages=False)
        if existing is None:
            self.client.create_chat(chat.thread_id, agent_id=chat.agent_id, title=chat.title)
            return
        self.client.update_chat(
            chat.thread_id,
            title=chat.title,
            summary=chat.summary,
        )

    def get_chat(self, thread_id: str) -> Chat | None:
        payload = self.client.get_chat(thread_id, include_messages=False)
        return _chat_from_payload(payload) if payload is not None else None

    def list_chats(self, limit: int = 50, *, agent_id: str | None = None) -> list[Chat]:
        return [
            _chat_from_payload(payload)
            for payload in self.client.list_chats(limit, agent_id=agent_id)
        ]

    def load_chat_with_messages(self, thread_id: str) -> tuple[Chat, list[ChatMessage]] | None:
        """Fetch metadata and history in the single request that carries both."""

        payload = self.client.get_chat(thread_id)
        if payload is None:
            return None
        messages = [
            ChatMessage.model_validate(message) for message in payload.get("messages", [])
        ]
        return _chat_from_payload(payload), messages

    def load_messages(self, thread_id: str) -> list[ChatMessage]:
        loaded = self.load_chat_with_messages(thread_id)
        return loaded[1] if loaded is not None else []

    def delete_chat(self, thread_id: str) -> None:
        self.client.delete_thread(thread_id)


def get_chat_storage(client: BaseAgentClient) -> ChatStorage:
    """Create a session-local adapter for an authenticated API client.

    Streamlit module globals are shared by browser sessions, so the adapter
    must never cache a user's session token globally.
    """
    return ChatStorage(client)
