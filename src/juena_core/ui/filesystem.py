"""React browser for the shared conversation filesystem, behind core's ui extra."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

__all__ = ["render_filesystem"]
_ASSETS = Path(__file__).with_name("filesystem_dist")


@st.cache_resource
def _component():
    return st.components.v2.component(
        "juena_filesystem",
        js=(_ASSETS / "filesystem.js").read_text(),
        # An interior newline keeps minified CSS from being mistaken for a glob.
        css="/* Conversation explorer */\n" + (_ASSETS / "filesystem.css").read_text(),
    )


@st.fragment
def render_filesystem(*, api_url: str, thread_id: str, session_scope: str,
                      refresh_generation: int = 0) -> None:
    """Mount before consuming chat SSE; file requests run entirely in the browser.

    Only view bookkeeping and explicit composer handoffs cross the Python bridge.
    Its fragment reruns queue safely while the foreground stream is being consumed.
    """
    if not (_ASSETS / "filesystem.js").is_file():
        st.error("Build juena-core's explorer: cd frontend/filesystem && npm ci && npm run build")
        return
    if st.session_state.get("filesystem_scope") != session_scope:
        st.session_state.filesystem_scope = session_scope
        st.session_state.filesystem_views = {}
        st.session_state.filesystem_seen_events = set()
        st.session_state.pop("filesystem_mention", None)
    views = st.session_state.filesystem_views
    key = f"filesystem:{session_scope}:{thread_id}"

    def remember_view():
        value = st.session_state.get(key, {}).get("view")
        if value is not None:
            views[thread_id] = value

    def auth_lost():
        views.clear()
        st.session_state.pop("filesystem_mention", None)

    result = _component()(
        data={"apiUrl": api_url.rstrip("/"), "threadId": thread_id,
              "sessionScope": session_scope, "refreshGeneration": refresh_generation,
              "initialView": views.get(thread_id)},
        key=key, default={"view": views.get(thread_id), "ask": None},
        on_view_change=remember_view, on_ask_change=lambda: None,
        on_auth_lost_change=auth_lost,
    )
    # Keep the event as component state until acknowledged: transient triggers
    # can be coalesced with view updates while Python is consuming SSE.
    event = result.ask
    seen = st.session_state.filesystem_seen_events
    if event and event.get("thread_id") == st.session_state.get("thread_id") and event["id"] not in seen:
        seen.add(event["id"])
        st.session_state.filesystem_mention = event
        st.rerun(scope="app")
