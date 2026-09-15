"""Stub for 01/CP4. Ported from ``juena/server/utils.py`` (00-BOUNDARY.md,
*Moves whole*)."""

from __future__ import annotations

from langchain_core.messages import BaseMessage

from juena_core.schema.server import ChatMessage

__all__ = ["message_text", "reasoning_from", "display_text_from", "display_attachments_from", "langchain_to_chat_message"]


def message_text(message: BaseMessage) -> str:
    raise NotImplementedError("juena_core.server.utils.message_text lands in 01/CP4")


def reasoning_from(message: BaseMessage) -> str:
    raise NotImplementedError("juena_core.server.utils.reasoning_from lands in 01/CP4")


def display_text_from(message: BaseMessage) -> str:
    raise NotImplementedError("juena_core.server.utils.display_text_from lands in 01/CP4")


def display_attachments_from(message: BaseMessage) -> list[dict[str, object]]:
    raise NotImplementedError("juena_core.server.utils.display_attachments_from lands in 01/CP4")


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    raise NotImplementedError("juena_core.server.utils.langchain_to_chat_message lands in 01/CP4")
