"""Small Streamlit page demonstrating the reusable UI and client seams."""

from __future__ import annotations

import os
from uuid import uuid4

import streamlit as st

from juena_core.clients.base import AgentClientError
from juena_core.schema.server import ChatMessage
from juena_core.ui.client_setup import initialize_client
from juena_core.ui.components import render_message, reset_rendered_artifacts
from juena_core.ui.streaming import stream_and_display_response

from examples.simple_chat.agent import DEMO_AGENT_ID
from examples.simple_chat.config import build_settings

API_URL = os.getenv("DEMO_API_URL", "http://127.0.0.1:8080")
DEMO_SETTINGS = build_settings()
PROVIDER = DEMO_SETTINGS.DEFAULT_PROVIDER
MODEL = DEMO_SETTINGS.DEFAULT_MODEL


def _new_chat() -> None:
    """Create a persisted thread and reset this browser session's transcript."""

    thread_id = str(uuid4())
    st.session_state.client.create_chat(
        thread_id,
        agent_id=DEMO_AGENT_ID,
        title="Demo chat",
    )
    st.session_state.thread_id = thread_id
    st.session_state.messages = []
    st.session_state.pending_approvals = {}
    st.session_state.approval_checked_threads = set()


st.set_page_config(page_title="juena-core demo", page_icon="💬")

if "client" not in st.session_state:
    st.session_state.client = initialize_client(
        API_URL,
        DEMO_AGENT_ID,
        timeout=60,
    )
    st.session_state.selected_provider = PROVIDER
    st.session_state.selected_model = MODEL
    st.session_state.pending_approvals = {}
    st.session_state.approval_checked_threads = set()
    st.session_state.messages = []

if not st.session_state.client.health():
    st.error(f"The demo API is not reachable at {API_URL}.")
    st.stop()

try:
    if "thread_id" not in st.session_state:
        _new_chat()
except AgentClientError as exc:
    st.error(f"Could not create a demo chat: {exc}")
    st.stop()

reset_rendered_artifacts()
st.title("juena-core simple chat")
st.caption(f"Agent: `{DEMO_AGENT_ID}` · Provider: `{PROVIDER}` · Model: `{MODEL}`")

if st.sidebar.button("New chat", use_container_width=True):
    try:
        _new_chat()
    except AgentClientError as exc:
        st.sidebar.error(str(exc))
    else:
        st.rerun()

for message in st.session_state.messages:
    render_message(message)

if prompt := st.chat_input("Ask something"):
    user_message = ChatMessage(type="human", content=prompt)
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        stream_and_display_response(prompt, st.empty())
