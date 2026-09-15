"""Stub for 01/CP4. Ported from ``juena/server/errors.py`` (00-BOUNDARY.md,
*Moves whole*)."""

from __future__ import annotations

__all__ = ["ChatbotServerError", "AgentNotFoundError", "StreamingError", "StateError", "MessageProcessingError"]


class ChatbotServerError(Exception):
    """Stub — implemented in 01/CP4."""


class AgentNotFoundError(ChatbotServerError):
    """Stub — implemented in 01/CP4."""


class StreamingError(ChatbotServerError):
    """Stub — implemented in 01/CP4."""


class StateError(ChatbotServerError):
    """Stub — implemented in 01/CP4."""


class MessageProcessingError(ChatbotServerError):
    """Stub — implemented in 01/CP4."""
