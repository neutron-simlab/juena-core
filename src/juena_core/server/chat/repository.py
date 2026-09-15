"""User-owned chat persistence operations.

Ownership metadata lives in the ``chats`` table. Message history is *not*
duplicated here: the LangGraph checkpointer already stores every message for a
thread, so :func:`load_thread_messages` reads it back from there.

Every lookup is scoped by owner **and** by agent. A thread belongs to one
registered graph, and resuming it under another would hand a checkpoint to a
graph whose state channels do not match it. Both mismatches raise
:class:`ChatNotFoundError` rather than a distinct "forbidden" error, because
telling a caller that a row exists but is not theirs is itself an answer.

:func:`load_thread_messages` is the one function here that reads *content*
rather than ownership. It goes to the checkpointer, not to a table: the
``chats`` row carries who owns a conversation, never what was said in it.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from juena_core.config import settings
from juena_core.log import get_logger
from juena_core.schema.server import ChatMessage
from juena_core.server.database.checkpointer import get_checkpointer
from juena_core.server.database.models import Chat
from juena_core.server.utils import langchain_to_chat_message

logger = get_logger(__name__)

__all__ = [
    "ChatNotFoundError",
    "ensure_owned_chat",
    "get_owned_chat",
    "list_owned_chats",
    "load_thread_messages",
    "chat_to_dict",
]


class ChatNotFoundError(LookupError):
    """Raised for missing chats, and for chats owned by another user or agent."""


async def ensure_owned_chat(
    session: AsyncSession,
    user_id: UUID,
    thread_id: str | None,
    agent_id: str,
) -> Chat:
    """Create a chat or verify ownership without revealing other users' rows."""

    effective_thread_id = thread_id or str(uuid4())
    chat = await session.get(Chat, effective_thread_id)
    if chat is None:
        chat = Chat(thread_id=effective_thread_id, user_id=user_id, agent_id=agent_id)
        session.add(chat)
        await session.flush()
    elif chat.user_id != user_id or chat.agent_id != agent_id:
        raise ChatNotFoundError(effective_thread_id)
    return chat


async def get_owned_chat(
    session: AsyncSession,
    user_id: UUID,
    thread_id: str,
    *,
    agent_id: str | None,
) -> Chat:
    """Load one chat the caller owns, requiring a matching agent when given one.

    ``agent_id`` accepts ``None`` — renaming a conversation is addressed by
    thread alone, and those routes carry no agent in their path — but it has
    **no default**, so every call site states which it is. A default would let
    a route that runs a graph skip the check by omission, which is the
    cross-agent resume this column exists to prevent, and it would pass every
    test that did not probe for it specifically.
    """

    conditions = [Chat.thread_id == thread_id, Chat.user_id == user_id]
    if agent_id is not None:
        conditions.append(Chat.agent_id == agent_id)
    result = await session.execute(select(Chat).where(*conditions))
    chat = result.scalar_one_or_none()
    if chat is None:
        raise ChatNotFoundError(thread_id)
    return chat


async def list_owned_chats(
    session: AsyncSession,
    user_id: UUID,
    *,
    agent_id: str | None = None,
    limit: int = 50,
) -> list[Chat]:
    """List the caller's chats, newest first, optionally for one agent only.

    Here ``agent_id`` *does* default to ``None``: listing everything shows too
    much rather than granting anything, so an omission is a display choice,
    not a hole.
    """

    conditions = [Chat.user_id == user_id]
    if agent_id is not None:
        conditions.append(Chat.agent_id == agent_id)
    result = await session.execute(
        select(Chat)
        .where(*conditions)
        .order_by(Chat.updated_at.desc())
        .limit(limit)
    )
    return list(result.scalars())


async def load_thread_messages(thread_id: str) -> list[ChatMessage]:
    """Read a thread's message history from the LangGraph checkpointer.

    The checkpointer is the single source of truth for conversation content.
    Callers must verify ownership of ``thread_id`` before calling this — this
    function takes no user, and would happily read anyone's thread.
    """

    checkpoint = await get_checkpointer().aget({"configurable": {"thread_id": thread_id}})
    if checkpoint is None:
        return []

    include_tool_payloads = settings().STREAM_TOOL_PAYLOADS
    messages: list[ChatMessage] = []
    for message in checkpoint.get("channel_values", {}).get("messages", []):
        if not isinstance(message, BaseMessage) or isinstance(message, SystemMessage):
            continue
        # Tool results are activity, not transcript. The live stream reports
        # them as compact status events rather than messages, so including them
        # here would make a reloaded thread render differently from how the
        # user watched it happen.
        if isinstance(message, ToolMessage) and not include_tool_payloads:
            continue
        try:
            chat_message = langchain_to_chat_message(message)
        except ValueError:
            logger.warning("Skipping unsupported checkpointed message in thread %s", thread_id)
            continue
        # Tool-call announcements have no content of their own; the live stream
        # drops them too.
        if (
            chat_message.type == "ai"
            and chat_message.tool_calls
            and not (chat_message.content or "").strip()
        ):
            continue
        chat_message.thread_id = thread_id
        messages.append(chat_message)
    return messages


def chat_to_dict(chat: Chat) -> dict[str, str]:
    """Serialise a chat for the API, including the agent that owns it.

    ``agent_id`` is part of the payload so a UI can restore the mode a
    conversation belongs to instead of inferring it from whatever is selected.
    """

    return {
        "thread_id": chat.thread_id,
        "agent_id": chat.agent_id,
        "title": chat.title,
        "summary": chat.summary,
        "created_at": chat.created_at.isoformat(),
        "updated_at": chat.updated_at.isoformat(),
    }
