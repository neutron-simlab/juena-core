"""Stub for 01/CP4. Ported from ``juena/server/chat/endpoints.py``
(00-BOUNDARY.md, *Moves whole*). ``create_chat`` and ``list_chats`` carry
the ``agent_id`` requirement from 01/CP3's write-and-read path."""

from __future__ import annotations

from typing import Any

__all__ = ["list_chats", "create_chat", "get_chat", "update_chat"]


async def list_chats(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.server.chat.endpoints.list_chats lands in 01/CP4")


async def create_chat(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.server.chat.endpoints.create_chat lands in 01/CP4")


async def get_chat(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.server.chat.endpoints.get_chat lands in 01/CP4")


async def update_chat(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.server.chat.endpoints.update_chat lands in 01/CP4")
