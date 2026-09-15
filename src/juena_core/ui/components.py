"""Stub for 01/CP5. Split out of ``juena-chatbot/app/ui_components.py``
(00-BOUNDARY.md, decision 7). ``logo_data_uri`` and
``render_header_with_logo`` stay in each application's own UI — a logo is
domain, not wiring."""

from __future__ import annotations

from typing import Any

__all__ = [
    "sanitize_assistant_content",
    "render_content",
    "render_message",
    "render_streaming_token",
    "render_artifacts",
    "finalize_streaming_message",
    "render_attachment_chips",
    "should_collapse_tool_payload",
]


def sanitize_assistant_content(content: Any) -> Any:
    raise NotImplementedError("juena_core.ui.components.sanitize_assistant_content lands in 01/CP5")


def render_content(content: Any) -> None:
    raise NotImplementedError("juena_core.ui.components.render_content lands in 01/CP5")


def render_message(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("juena_core.ui.components.render_message lands in 01/CP5")


def render_streaming_token(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("juena_core.ui.components.render_streaming_token lands in 01/CP5")


def render_artifacts(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("juena_core.ui.components.render_artifacts lands in 01/CP5")


def finalize_streaming_message(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.ui.components.finalize_streaming_message lands in 01/CP5")


def render_attachment_chips(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("juena_core.ui.components.render_attachment_chips lands in 01/CP5")


def should_collapse_tool_payload(*args: Any, **kwargs: Any) -> bool:
    raise NotImplementedError("juena_core.ui.components.should_collapse_tool_payload lands in 01/CP5")
