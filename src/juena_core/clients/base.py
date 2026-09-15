"""Stub for 01/CP5. Split out of ``juena/clients/client.py`` (00-BOUNDARY.md,
decision 10) — ``AgentClient`` calls ``/auth/me`` and ``/research``, neither
of which is a core route, so core gets only ``BaseAgentClient``: transport,
SSE parsing, and the methods for routes core owns. juena-chatbot and v2 each
subclass it with whatever their own application adds.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Any

from juena_core.schema.llm_models import Provider
from juena_core.schema.server import ChatMessage

__all__ = ["AgentClientError", "AgentAuthenticationError", "BaseAgentClient"]


class AgentClientError(Exception):
    """Stub — implemented in 01/CP5."""


class AgentAuthenticationError(AgentClientError):
    """Stub — implemented in 01/CP5."""


def _http_error_message(exc: Exception) -> str:
    raise NotImplementedError("juena_core.clients.base._http_error_message lands in 01/CP5")


class BaseAgentClient:
    """Stub — implemented in 01/CP5. Owns: ``health``, ``list_chats``,
    ``create_chat``, ``get_chat``, ``update_chat``, ``stream``,
    ``resume_stream``, ``get_artifact``, ``get_pending_interrupt``,
    ``delete_thread``. Does **not** own ``get_current_user`` or
    ``list_research`` — those are application routes."""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        session_token: str | None = None,
    ) -> None:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.__init__ lands in 01/CP5")

    def close(self) -> None:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.close lands in 01/CP5")

    def __enter__(self) -> "BaseAgentClient":
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.__enter__ lands in 01/CP5")

    def __exit__(self, *_exc_info: object) -> None:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.__exit__ lands in 01/CP5")

    def health(self) -> bool:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.health lands in 01/CP5")

    @property
    def _headers(self) -> dict[str, str]:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient._headers lands in 01/CP5")

    def _build_attachment_payloads(
        self,
        attachments: Sequence[Any] | None,
    ) -> list[tuple[str, tuple[str, bytes, str]]]:
        raise NotImplementedError(
            "juena_core.clients.base.BaseAgentClient._build_attachment_payloads lands in 01/CP5"
        )

    def _build_file_request_data(
        self,
        message: str,
        model: str | None,
        provider: str | Provider | None,
        thread_id: str | None,
    ) -> dict[str, str]:
        raise NotImplementedError(
            "juena_core.clients.base.BaseAgentClient._build_file_request_data lands in 01/CP5"
        )

    def list_chats(self, limit: int = 50, *, agent_id: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.list_chats lands in 01/CP5")

    def create_chat(
        self,
        thread_id: str,
        *,
        agent_id: str,
        title: str = "New Chat",
    ) -> dict[str, Any]:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.create_chat lands in 01/CP5")

    def get_chat(self, thread_id: str, include_messages: bool = True) -> dict[str, Any] | None:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.get_chat lands in 01/CP5")

    def update_chat(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.update_chat lands in 01/CP5")

    def _parse_sse_data(self, data: str) -> ChatMessage | dict[str, Any] | None:
        raise NotImplementedError(
            "juena_core.clients.base.BaseAgentClient._parse_sse_data lands in 01/CP5"
        )

    def _build_stream_request(
        self,
        message: str,
        model: str | None,
        provider: str | None,
        thread_id: str | None,
        attachments: Sequence[Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError(
            "juena_core.clients.base.BaseAgentClient._build_stream_request lands in 01/CP5"
        )

    def stream(
        self,
        message: str,
        model: str | None = None,
        provider: str | None = None,
        thread_id: str | None = None,
        attachments: Sequence[Any] | None = None,
    ) -> Generator[ChatMessage | dict[str, Any], None, None]:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.stream lands in 01/CP5")
        yield  # pragma: no cover - preserve the synchronous generator contract

    def resume_stream(
        self,
        *,
        thread_id: str,
        interrupt_id: str,
        decision: str | None = None,
        edited_command: str | None = None,
        answer: str | None = None,
        model: str | None = None,
        provider: str | Provider | None = None,
    ) -> Generator[ChatMessage | dict[str, Any], None, None]:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.resume_stream lands in 01/CP5")
        yield  # pragma: no cover - preserve the synchronous generator contract

    def get_artifact(self, artifact_id: str) -> bytes:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.get_artifact lands in 01/CP5")

    def get_pending_interrupt(
        self,
        thread_id: str,
        *,
        model: str | None = None,
        provider: str | Provider | None = None,
    ) -> dict[str, Any] | None:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.get_pending_interrupt lands in 01/CP5")

    def is_token_message(self, message: ChatMessage | str | dict[str, Any]) -> bool:
        raise NotImplementedError(
            "juena_core.clients.base.BaseAgentClient.is_token_message lands in 01/CP5"
        )

    def get_token_content(self, message: ChatMessage | str | dict[str, Any]) -> str | None:
        raise NotImplementedError(
            "juena_core.clients.base.BaseAgentClient.get_token_content lands in 01/CP5"
        )

    def delete_thread(self, thread_id: str) -> dict[str, Any]:
        raise NotImplementedError("juena_core.clients.base.BaseAgentClient.delete_thread lands in 01/CP5")
