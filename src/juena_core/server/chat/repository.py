"""User-owned chat persistence operations.

The ``chats`` table holds who owns a conversation, never what was said in it.
The LangGraph checkpointer already stores every message of a thread, so
:func:`load_thread_messages` reads history from there.

Lookups are scoped by owner, and by agent when one is given. A thread belongs
to one registered graph, and resuming it under another would hand that graph a
checkpoint that does not fit its state. Both mismatches raise
:class:`ChatNotFoundError`, not a "forbidden" error, because saying that a chat
exists but is not the caller's already tells them something.
"""

from __future__ import annotations

import re
from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from juena_core.config import settings
from juena_core.filesystem.service import ConversationFilesystem
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


#: What a client may name a new conversation: letters, digits, ``-`` and ``_``,
#: which covers the UUIDs the UI generates. The checkpointer keys state by this
#: name, and an application may also checkpoint work no user owns under
#: prefixed names (such as ``research:<job id>``), so a new conversation must
#: not be able to spell one of those.
_NEW_THREAD_ID = re.compile(r"[A-Za-z0-9_-]{1,255}")


class ChatNotFoundError(LookupError):
    """Raised for missing chats, and for chats owned by another user or agent."""


class ChatDeletedError(ChatNotFoundError):
    """A retired workspace identity must not be reused by a stale tab."""


async def ensure_owned_chat(
    session: AsyncSession,
    user_id: UUID,
    thread_id: str | None,
    agent_id: str,
    *, filesystem: ConversationFilesystem | None = None,
) -> Chat:
    """Create a chat or verify ownership without revealing other users' rows.

    Args:
        session: The database session. A new row is flushed, not committed.
        user_id: The caller's id.
        thread_id: The conversation thread. None creates a new one.
        agent_id: The agent the request addresses.
        filesystem: Shared persistent conversation files, used to refuse a
            deleted conversation.

    Returns:
        The caller's chat row.

    Raises:
        ChatDeletedError: The conversation was deleted.
        ChatNotFoundError: The chat belongs to another user or agent, or a new
            thread id is not an allowed name.
    """

    effective_thread_id = thread_id or str(uuid4())
    if filesystem is not None:
        files = filesystem.for_thread(str(user_id), effective_thread_id)
        if files.store.deleted(files.workspace_id):
            raise ChatDeletedError(effective_thread_id)
    chat = await session.get(Chat, effective_thread_id)
    if chat is None:
        # Checked only when creating: a conversation that already exists keeps
        # working whatever it was named.
        if not _NEW_THREAD_ID.fullmatch(effective_thread_id):
            raise ChatNotFoundError(effective_thread_id)
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

    Args:
        session: The database session.
        user_id: The caller's id.
        thread_id: The conversation thread.
        agent_id: The agent the chat must belong to. ``None`` skips that check,
            for routes addressed by thread alone, such as renaming a
            conversation. It has no default, so every call site states which
            it means: a default would let a route that runs a graph skip the
            agent check by omission.

    Returns:
        The chat row.

    Raises:
        ChatNotFoundError: The caller owns no such chat, or it belongs to
            another agent.
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

    Args:
        session: The database session.
        user_id: The caller's id.
        agent_id: Only this agent's chats. Here it defaults to ``None`` (every
            agent): that only shows more of the caller's own rows, and grants
            nothing.
        limit: The most chats to return.

    Returns:
        The chat rows, most recently updated first.
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

    Callers must check ownership of ``thread_id`` first: this function takes
    no user and reads anyone's thread.

    Args:
        thread_id: The conversation thread.

    Returns:
        The transcript as the UI shows it; empty when the thread has no
        checkpoint.
    """

    checkpoint = await get_checkpointer().aget({"configurable": {"thread_id": thread_id}})
    if checkpoint is None:
        return []

    include_tool_payloads = settings().STREAM_TOOL_PAYLOADS
    messages: list[ChatMessage] = []
    values = checkpoint.get("channel_values", {})
    # Undelivered corrections are transcript entries, never model history.
    receipts = {}
    for receipt in values.get("unapplied_steering", []):
        anchor = receipt.additional_kwargs["steering"].get("after_message_id")
        receipts.setdefault(anchor, []).append(receipt)
    transcript = []
    for message in values.get("messages", []):
        transcript.append(message)
        transcript.extend(receipts.pop(message.id, []))
    # A summarizer may have evicted the anchor; keep its receipts in order.
    for remaining in receipts.values():
        transcript.extend(remaining)
    for message in transcript:
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

    Args:
        chat: The chat row.

    Returns:
        The chat's thread id, agent, title, summary and timestamps.
    """

    return {
        "thread_id": chat.thread_id,
        "agent_id": chat.agent_id,
        "title": chat.title,
        "summary": chat.summary,
        "created_at": chat.created_at.isoformat(),
        "updated_at": chat.updated_at.isoformat(),
    }
