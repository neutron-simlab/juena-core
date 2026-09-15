"""Authenticated, user-owned chat history API.

Thread deletion lives on ``DELETE /threads/{thread_id}`` in
:mod:`juena_core.server.api.endpoints`; it is not duplicated here.

The router is built by a factory rather than declared at module scope, because
the identity dependency is the application's: juena-chatbot passes a SAML-backed
one, VITESS v2 passes :func:`~juena_core.server.identity.local_principal`. The
dependency is then visible in each route's signature instead of hidden in a
dictionary mutation at startup.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from juena_core.schema.server import CreateChatInput, UpdateChatInput
from juena_core.server.chat.repository import (
    ChatNotFoundError,
    chat_to_dict,
    ensure_owned_chat,
    get_owned_chat,
    list_owned_chats,
    load_thread_messages,
)
from juena_core.server.database.connection import get_db_session
from juena_core.server.identity import Principal, PrincipalDependency

__all__ = ["build_chat_router"]


def build_chat_router(principal: PrincipalDependency) -> APIRouter:
    """Build the ``/chats`` router against one application's identity provider."""

    router = APIRouter(prefix="/chats", tags=["chats"])

    @router.get("")
    async def list_chats(
        agent_id: Annotated[str | None, Query(max_length=64)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        user: Principal = Depends(principal),
        session: AsyncSession = Depends(get_db_session),
    ) -> list[dict]:
        """List the caller's conversations, newest first.

        ``agent_id`` narrows the list to one agent. Omitting it lists them all,
        which shows more rather than granting more — every row is already the
        caller's own.
        """

        chats = await list_owned_chats(session, user.id, agent_id=agent_id, limit=limit)
        return [chat_to_dict(chat) for chat in chats]

    @router.post("", status_code=status.HTTP_201_CREATED)
    async def create_chat(
        payload: CreateChatInput,
        user: Principal = Depends(principal),
        session: AsyncSession = Depends(get_db_session),
    ) -> dict:
        try:
            chat = await ensure_owned_chat(session, user.id, payload.thread_id, payload.agent_id)
        except ChatNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        chat.title = payload.title.strip() or "New Chat"
        await session.commit()
        return chat_to_dict(chat)

    @router.get("/{thread_id}")
    async def get_chat(
        thread_id: str,
        include_messages: Annotated[bool, Query()] = True,
        user: Principal = Depends(principal),
        session: AsyncSession = Depends(get_db_session),
    ) -> dict:
        """Return chat metadata, and by default its checkpointed message history.

        Pass ``include_messages=false`` for existence and title checks so callers
        do not pay for a full history read they will discard.

        No agent is required here: the path carries none, reading a conversation
        the caller owns is safe under any agent, and the response says which
        agent it belongs to.
        """

        try:
            chat = await get_owned_chat(session, user.id, thread_id, agent_id=None)
        except ChatNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        payload = chat_to_dict(chat)
        if include_messages:
            payload["messages"] = await load_thread_messages(chat.thread_id)
        return payload

    @router.patch("/{thread_id}")
    async def update_chat(
        thread_id: str,
        payload: UpdateChatInput,
        user: Principal = Depends(principal),
        session: AsyncSession = Depends(get_db_session),
    ) -> dict:
        try:
            chat = await get_owned_chat(session, user.id, thread_id, agent_id=None)
        except ChatNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        if payload.title is not None:
            chat.title = payload.title.strip() or "New Chat"
        if payload.summary is not None:
            chat.summary = payload.summary
        await session.commit()
        return chat_to_dict(chat)

    return router
