"""Stub for 01/CP3. Ported from ``juena/server/chat/repository.py``
(00-BOUNDARY.md, *Moves whole*).

``ensure_owned_chat`` gains ``agent_id`` (00-BOUNDARY.md, decision 13): sets
it on creation, and raises ``ChatNotFoundError`` when an existing chat's
``agent_id`` differs — the same shape as the existing owner check, and for
the same reason: a mismatch must not reveal that the row exists.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from juena_core.schema.server import ChatMessage
from juena_core.server.database.models import Chat

__all__ = ["ChatNotFoundError", "ensure_owned_chat", "get_owned_chat", "list_owned_chats", "load_thread_messages", "chat_to_dict"]


class ChatNotFoundError(LookupError):
    """Stub — implemented in 01/CP3."""


async def ensure_owned_chat(
    session: AsyncSession,
    user_id: UUID,
    thread_id: str | None,
    agent_id: str,
    *args: Any,
    **kwargs: Any,
) -> Chat:
    raise NotImplementedError("juena_core.server.chat.repository.ensure_owned_chat lands in 01/CP3")


async def get_owned_chat(
    session: AsyncSession,
    user_id: UUID,
    thread_id: str,
    *,
    agent_id: str | None = None,
) -> Chat:
    raise NotImplementedError("juena_core.server.chat.repository.get_owned_chat lands in 01/CP3")


async def list_owned_chats(
    session: AsyncSession,
    user_id: UUID,
    *,
    agent_id: str | None = None,
    limit: int = 50,
) -> list[Chat]:
    raise NotImplementedError("juena_core.server.chat.repository.list_owned_chats lands in 01/CP3")


async def load_thread_messages(thread_id: str) -> list[ChatMessage]:
    raise NotImplementedError("juena_core.server.chat.repository.load_thread_messages lands in 01/CP3")


def chat_to_dict(chat: Chat) -> dict[str, str]:
    raise NotImplementedError("juena_core.server.chat.repository.chat_to_dict lands in 01/CP3")
