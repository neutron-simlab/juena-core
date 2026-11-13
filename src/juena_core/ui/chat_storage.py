"""Authenticated chat-history adapter for the Streamlit UI.

The FastAPI service owns the Postgres history. Streamlit keeps no local
conversation database and goes through this small synchronous adapter.

:class:`Chat` carries ``agent_id``, so restoring a conversation restores the
agent it belongs to, not whichever one the page has selected. Opening a thread
under the wrong agent answers 404 on every later request.
:meth:`ChatStorage.list_chats` takes an optional agent filter, so a UI that
serves several agents can show one agent's conversations at a time.
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
    """One user-owned conversation, as the sidebar knows it.

    Attributes:
        thread_id: The conversation's id.
        agent_id: The agent the conversation belongs to.
        title: The title shown in the sidebar.
        summary: A short summary, empty when there is none.
        created_at: When the conversation was created.
        updated_at: When it last changed.
    """

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
        """Initialize the adapter.

        Args:
            client: The signed-in user's API client.
        """
        self.client = client

    def upsert_chat(self, chat: Chat) -> None:
        """Create the conversation, or update its title and summary if it exists.

        Args:
            chat: The conversation to save.
        """
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
        """Fetch one conversation's metadata, without its messages.

        Args:
            thread_id: The conversation's id.

        Returns:
            The conversation, or None when the API has no such conversation.
        """
        payload = self.client.get_chat(thread_id, include_messages=False)
        return _chat_from_payload(payload) if payload is not None else None

    def list_chats(self, limit: int = 50, *, agent_id: str | None = None) -> list[Chat]:
        """List the user's conversations.

        Args:
            limit: The most conversations to return.
            agent_id: Return only this agent's conversations. None means all.

        Returns:
            The conversations, in the order the API sends them.
        """
        return [
            _chat_from_payload(payload)
            for payload in self.client.list_chats(limit, agent_id=agent_id)
        ]

    def load_chat_with_messages(self, thread_id: str) -> tuple[Chat, list[ChatMessage]] | None:
        """Fetch metadata and history in the single request that carries both.

        Args:
            thread_id: The conversation's id.

        Returns:
            The conversation and its messages, or None when it does not exist.
        """

        payload = self.client.get_chat(thread_id)
        if payload is None:
            return None
        messages = [
            ChatMessage.model_validate(message) for message in payload.get("messages", [])
        ]
        return _chat_from_payload(payload), messages

    def load_messages(self, thread_id: str) -> list[ChatMessage]:
        """Fetch one conversation's messages.

        Args:
            thread_id: The conversation's id.

        Returns:
            Its messages, or an empty list when it does not exist.
        """
        loaded = self.load_chat_with_messages(thread_id)
        return loaded[1] if loaded is not None else []

    def delete_chat(self, thread_id: str) -> None:
        """Delete a conversation.

        Args:
            thread_id: The conversation's id.
        """
        self.client.delete_thread(thread_id)


def get_chat_storage(client: BaseAgentClient) -> ChatStorage:
    """Create a session-local adapter for an authenticated API client.

    Streamlit module globals are shared by browser sessions, so the adapter
    must never cache a user's session token globally.

    Args:
        client: The signed-in user's API client.

    Returns:
        A new adapter for this browser session.
    """
    return ChatStorage(client)
