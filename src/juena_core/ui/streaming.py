"""Stub for 01/CP5. Split out of ``juena-chatbot/app/chat_interface.py``
(00-BOUNDARY.md, decision 7). The server-sent-event contract is core's; the
page is the app's — ``render_chat_interface()`` and
``render_starter_topics()`` stay in each application's own ``sidebar.py`` /
``starters.py``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "process_stream_chunk",
    "is_status_chunk",
    "is_approval_chunk",
    "is_interrupt_chunk",
    "is_artifact_chunk",
    "apply_status_chunk",
    "stream_and_display_response",
    "stream_and_display_resume",
    "render_pending_interrupt",
]


def process_stream_chunk(chunk: dict[str, Any]) -> Any:
    raise NotImplementedError("juena_core.ui.streaming.process_stream_chunk lands in 01/CP5")


def is_status_chunk(chunk: dict[str, Any]) -> bool:
    raise NotImplementedError("juena_core.ui.streaming.is_status_chunk lands in 01/CP5")


def is_approval_chunk(chunk: dict[str, Any]) -> bool:
    raise NotImplementedError("juena_core.ui.streaming.is_approval_chunk lands in 01/CP5")


def is_interrupt_chunk(chunk: dict[str, Any]) -> bool:
    raise NotImplementedError("juena_core.ui.streaming.is_interrupt_chunk lands in 01/CP5")


def is_artifact_chunk(chunk: dict[str, Any]) -> bool:
    raise NotImplementedError("juena_core.ui.streaming.is_artifact_chunk lands in 01/CP5")


def apply_status_chunk(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.ui.streaming.apply_status_chunk lands in 01/CP5")


def _stream_and_display_chunks(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.ui.streaming._stream_and_display_chunks lands in 01/CP5")


def stream_and_display_response(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.ui.streaming.stream_and_display_response lands in 01/CP5")


def stream_and_display_resume(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.ui.streaming.stream_and_display_resume lands in 01/CP5")


def _render_approval_card(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.ui.streaming._render_approval_card lands in 01/CP5")


def _render_question_card(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.ui.streaming._render_question_card lands in 01/CP5")


def render_pending_interrupt(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.ui.streaming.render_pending_interrupt lands in 01/CP5")
