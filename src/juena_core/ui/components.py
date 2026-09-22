"""Rendering for the message types core's stream produces.

Split out of ``juena-chatbot/app/ui_components.py`` (00-BOUNDARY.md,
decision 7). What moved is everything that renders a ``ChatMessage``, an
artifact or a streamed token — shapes core's server defines, so core is the
one place that has to agree with it.

What stayed behind is everything that says which product this is:
``logo_data_uri``, ``render_header_with_logo`` and ``render_chat_input_styles``
with their asset path and CSS. A logo is domain, not wiring.

**Artifacts are drawn once per script run.** One file can reach the page from
both the stored transcript and the live stream, and Streamlit treats a
repeated widget key as a fatal error rather than absorbing it, so
:func:`reset_rendered_artifacts` must be called at the top of every run.
"""

from __future__ import annotations

import base64
import binascii
import html
import json
import re
from typing import Any

import streamlit as st

from juena_core.schema.server import ChatMessage
from juena_core.ui.math_rendering import content_contains_math_markup, normalize_math_markdown

__all__ = [
    "RENDERED_ARTIFACTS_KEY",
    "PLOT_PREVIEW_WIDTH_PX",
    "sanitize_assistant_content",
    "render_content",
    "render_message",
    "render_streaming_token",
    "render_artifacts",
    "reset_rendered_artifacts",
    "finalize_streaming_message",
    "render_attachment_chips",
    "should_collapse_tool_payload",
]

PLOT_PREVIEW_WIDTH_PX = 720
# Artifact ids already given a widget key during the current script run.
RENDERED_ARTIFACTS_KEY = "rendered_artifact_keys"

_DATA_IMAGE_MARKER_RE = re.compile(
    r"data:image/[a-z0-9.+-]+;base64,",
    flags=re.IGNORECASE,
)
_BASE64_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)
# Every assistant image in these products is rendered from a validated artifact,
# so a Markdown image pointing anywhere else is a dead link -- most often a
# sandbox: path the model invented for a file that was never delivered. Stored
# threads still contain them, so this runs on history as well as new answers.
_UNRENDERABLE_IMAGE_RE = re.compile(
    r"!\[[^\]\n]*\]\((?!\s*https?://)\s*[^)\n]*\)",
    flags=re.IGNORECASE,
)
_DEAD_SCHEME_LINK_RE = re.compile(
    r"\[([^\]\n]*)\]\(\s*(?:sandbox|file):[^)\n]*\)",
    flags=re.IGNORECASE,
)


def sanitize_assistant_content(content: Any) -> Any:
    """Remove image markup Streamlit cannot resolve, preserving surrounding text.

    Generated images are rendered from validated artifacts, so any image the
    model writes inline is dead: an older turn may carry a very large Base64
    data URI, which is slow to send and renders broken, and a turn whose file
    was never delivered may carry an invented ``sandbox:`` path.
    """
    if not isinstance(content, str):
        return content

    cleaned = content
    search_from = 0
    while match := _DATA_IMAGE_MARKER_RE.search(cleaned, search_from):
        start = match.start()
        markdown_start = cleaned.rfind("![", max(0, start - 512), start)
        if markdown_start >= 0 and re.fullmatch(
            r"!\[[^\]\n]*\]\(\s*",
            cleaned[markdown_start:start],
        ):
            start = markdown_start

        end = match.end()
        while end < len(cleaned) and cleaned[end] in _BASE64_CHARACTERS:
            end += 1
        if end < len(cleaned) and cleaned[end] == ")":
            end += 1
        cleaned = cleaned[:start] + cleaned[end:]
        search_from = start

    # The scan above stays because it handles unterminated multi-megabyte blobs,
    # which a regex bounded to one line cannot match.
    cleaned = _UNRENDERABLE_IMAGE_RE.sub("", cleaned)
    cleaned = _DEAD_SCHEME_LINK_RE.sub(r"\1", cleaned)

    return re.sub(r"\n{3,}", "\n\n", cleaned)


def render_content(content: Any) -> None:
    """Render message content uniformly, as JSON or as markdown."""
    if isinstance(content, (dict, list)):
        st.json(content)
    else:
        try:
            parsed = json.loads(str(content))
            st.json(parsed)
        except (json.JSONDecodeError, TypeError):
            content_str = str(content) if content else ""
            if content_str.strip():
                st.markdown(normalize_math_markdown(content_str))


def should_collapse_tool_payload(custom_data: dict[str, Any] | None) -> bool:
    """Return True when tool payloads should render inside an expander."""
    custom_data = custom_data or {}
    display_mode = str(custom_data.get("display_mode", "")).strip().lower()
    if display_mode == "inline":
        return False
    if display_mode == "collapsed_by_default":
        return True
    # Legacy stored tool messages without custom_data should still collapse.
    return True


def _artifact_bytes(artifact: dict[str, Any]) -> bytes | None:
    """Resolve an artifact from a live SSE payload or its authenticated URL."""
    encoded = artifact.get("content_base64")
    if isinstance(encoded, str) and encoded:
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return None

    artifact_id = str(artifact.get("artifact_id") or "").strip()
    client = st.session_state.get("client")
    if not artifact_id or client is None:
        return None
    try:
        return client.get_artifact(artifact_id)
    except Exception:
        return None


@st.dialog("Plot preview", width="large", icon=":material/monitoring:")
def _render_plot_dialog(content: bytes, caption: str) -> None:
    """Show a generated PNG at the dialog's available width."""
    _render_png(content, caption, max_width="100%")


def _render_png(content: bytes, caption: str, *, max_width: str) -> None:
    """Render validated PNG bytes without a temporary Streamlit media URL."""
    encoded = base64.b64encode(content).decode("ascii")
    safe_caption = html.escape(caption, quote=True)
    st.html(
        f"""
        <figure style="margin:0;max-width:{max_width};width:100%;">
          <img
            src="data:image/png;base64,{encoded}"
            alt="{safe_caption}"
            style="display:block;max-width:100%;width:100%;height:auto;"
          >
          <figcaption style="margin-top:0.35rem;color:#6b7280;font-size:0.875rem;">
            {safe_caption}
          </figcaption>
        </figure>
        """
    )


def reset_rendered_artifacts() -> None:
    """Forget which artifacts have been drawn. Call once per script run."""
    st.session_state[RENDERED_ARTIFACTS_KEY] = set()


def _claim_artifact_key(artifact: dict[str, Any], filename: str) -> str | None:
    """Reserve the widget key suffix for *artifact* in this script run.

    Returns ``None`` when the same artifact has already been drawn: one file is
    worth showing once, and re-using its key is an error Streamlit raises rather
    than absorbs. Artifacts the server sent without an id cannot be recognised
    across renders, so they get a fresh suffix instead of being dropped.
    """
    drawn = st.session_state.get(RENDERED_ARTIFACTS_KEY)
    if not isinstance(drawn, set):
        drawn = set()
        st.session_state[RENDERED_ARTIFACTS_KEY] = drawn
    artifact_id = str(artifact.get("artifact_id") or "")
    key = artifact_id or f"{filename}#{len(drawn)}"
    if key in drawn:
        return None
    drawn.add(key)
    return key


def render_artifacts(custom_data: dict[str, Any] | None) -> None:
    """Render plots and authenticated downloads beneath the assistant response."""
    artifacts = (custom_data or {}).get("artifacts") or []
    if not isinstance(artifacts, list):
        return

    ordered = sorted(
        (item for item in artifacts if isinstance(item, dict)),
        key=lambda item: 0 if item.get("kind") == "image" else 1,
    )

    prepared: list[tuple[dict[str, Any], str, str]] = []
    for artifact in ordered:
        filename = str(artifact.get("filename") or "result.txt")
        key = _claim_artifact_key(artifact, filename)
        if key is not None:
            prepared.append((artifact, filename, key))

    sections: list[
        tuple[str | None, str | None, list[tuple[dict[str, Any], str, str]]]
    ] = []
    grouped_section: dict[str, int] = {}
    ungrouped_downloads = sum(
        not str(item[0].get("group_id") or "").strip()
        and item[0].get("kind") != "image"
        for item in prepared
    )
    for item in prepared:
        artifact = item[0]
        group_id = str(artifact.get("group_id") or "").strip()
        group_label = str(artifact.get("group_label") or "Artifacts").strip()
        if not group_id and artifact.get("kind") != "image" and ungrouped_downloads > 1:
            group_id = "__ungrouped_downloads__"
            group_label = "Generated files"
        if not group_id:
            sections.append((None, None, [item]))
            continue
        if group_id in grouped_section:
            sections[grouped_section[group_id]][2].append(item)
            continue
        grouped_section[group_id] = len(sections)
        sections.append((group_id, group_label or "Artifacts", [item]))

    for group_id, group_label, items in sections:
        if group_id is None:
            _render_artifact(*items[0])
            continue
        count = len(items)
        noun = "file" if count == 1 else "files"
        with st.expander(f"{group_label} ({count} {noun})", expanded=False):
            for item in items:
                _render_artifact(*item)


def _render_artifact(artifact: dict[str, Any], filename: str, key: str) -> None:
    """Render one already-claimed artifact."""

    content = _artifact_bytes(artifact)
    if not content:
        st.caption("This generated file is no longer available.")
        return
    caption = str(artifact.get("caption") or artifact.get("filename") or "Plot")
    mime_type = str(artifact.get("mime_type") or "application/octet-stream")
    if artifact.get("kind") == "image":
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            st.caption("This generated plot is no longer available.")
            return
        _render_png(content, caption, max_width=f"{PLOT_PREVIEW_WIDTH_PX}px")
        with st.container(horizontal=True):
            if st.button(
                "Expand plot",
                icon=":material/open_in_full:",
                help="Open a larger plot preview",
                key=f"artifact-expand:{key}",
            ):
                _render_plot_dialog(content, caption)
            st.download_button(
                f"Download {filename}",
                data=content,
                file_name=filename,
                mime=mime_type,
                icon=":material/download:",
                key=f"artifact-download:{key}",
            )
    else:
        st.caption(caption)
        st.download_button(
            f"Download {filename}",
            data=content,
            file_name=filename,
            mime=mime_type,
            icon=":material/download:",
            key=f"artifact-download:{key}",
        )


def finalize_streaming_message(
    message_placeholder: Any,
    content: Any,
    custom_data: dict[str, Any] | None = None,
) -> None:
    """Render the final AI message using the same content renderer as history."""
    with message_placeholder.container():
        if content:
            render_content(sanitize_assistant_content(content))
        render_artifacts(custom_data)


def render_attachment_chips(custom_data: dict[str, Any] | None) -> None:
    """Show the files attached to a user message, if any.

    The server records the same ``attachments`` key on the staged human message,
    so a reloaded chat shows the uploads it was sent with instead of losing every
    trace of them.
    """
    attachments = (custom_data or {}).get("attachments") or []
    if not isinstance(attachments, list):
        return

    labels: list[str] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        name = str(attachment.get("name") or "").strip()
        if not name:
            continue
        chars = attachment.get("chars")
        labels.append(
            f"📎 `{name}` ({chars} chars)" if isinstance(chars, int) else f"📎 `{name}`"
        )

    if labels:
        st.caption(" · ".join(labels))


def render_message(message: ChatMessage, show_system: bool = False) -> None:
    """Render one chat message with consistent styling.

    Args:
        message: The message to display.
        show_system: Whether to display system messages at all.
    """
    if message.type == "system" and not show_system:
        return

    if message.type == "human":
        with st.chat_message("user"):
            if content_contains_math_markup(message.content):
                render_content(message.content)
            else:
                st.write(message.content)
            render_attachment_chips(message.custom_data)

    elif message.type == "ai":
        with st.chat_message("assistant"):
            if message.content:
                render_content(sanitize_assistant_content(message.content))
            render_artifacts(message.custom_data)

    elif message.type == "tool":
        with st.chat_message("assistant"):
            tool_name = str((message.custom_data or {}).get("tool_name") or "").strip()
            label = f"Tool Result · `{tool_name}`" if tool_name else "Tool Result"
            st.markdown(f"🔧 **{label}**")
            if message.tool_call_id:
                st.caption(f"Tool call ID: {message.tool_call_id}")
            if message.content:
                if should_collapse_tool_payload(message.custom_data):
                    with st.expander("View tool payload", expanded=False):
                        render_content(message.content)
                else:
                    render_content(message.content)

    elif message.type == "system":
        with st.chat_message("assistant"):
            st.text(message.content)

    elif message.type == "custom":
        with st.chat_message("assistant"):
            if message.content:
                render_content(message.content)


def render_streaming_token(response_text: str, message_placeholder: Any) -> None:
    """Render the accumulated answer so far, with a trailing cursor."""
    message_placeholder.markdown(f"{sanitize_assistant_content(response_text)}▌")
