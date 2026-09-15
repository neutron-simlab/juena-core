"""Server exception types shared by the routers, the registry and streaming.

Each carries a ``message`` attribute alongside the usual ``str(exc)``, because
the streaming paths put that text on the wire and need it without the class
name or the details dict attached.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ChatbotServerError",
    "AgentNotFoundError",
    "StreamingError",
    "StateError",
    "MessageProcessingError",
]


class ChatbotServerError(Exception):
    """Base exception for all server errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AgentNotFoundError(ChatbotServerError):
    """Raised when a requested agent is not registered or cannot be created."""

    def __init__(self, agent_id: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"Agent '{agent_id}' not found or could not be created", details)
        self.agent_id = agent_id


class StreamingError(ChatbotServerError):
    """Raised when an error occurs during streaming operations."""

    def __init__(
        self,
        message: str,
        stream_mode: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.stream_mode = stream_mode


class StateError(ChatbotServerError):
    """Raised when an error occurs while preparing or reading graph state."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.operation = operation


class MessageProcessingError(ChatbotServerError):
    """Raised when one message cannot be converted for the client."""

    def __init__(
        self,
        message: str,
        message_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.message_type = message_type
