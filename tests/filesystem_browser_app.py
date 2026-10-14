"""Streamlit shell exercising core's actual component and blocking SSE consumer."""
from uuid import uuid4
import streamlit as st
from filesystem_browser_backend import API, setup
from juena_core.ui.filesystem import render_filesystem
from juena_core.ui.streaming import stream_and_display_response

st.set_page_config(layout="wide")
client = setup()
if "thread_id" not in st.session_state:
    st.session_state.update(client=client, thread_id="one", messages=[],
                            selected_provider="openai", selected_model="gpt-4o-mini", scope=str(uuid4()))
with st.sidebar:
    chosen = st.selectbox("Conversation", ["one", "two", "new"])
    if chosen != st.session_state.thread_id:
        st.session_state.thread_id = chosen
        st.session_state.messages = []
    if st.button("New session"):
        st.session_state.scope = str(uuid4())
    st.button("Rerun page")
chat, files = st.columns([4, 1], gap="medium")
with files:
    render_filesystem(api_url=API, thread_id=st.session_state.thread_id,
                      session_scope=st.session_state.scope,
                      refresh_generation=st.session_state.get("filesystem_refresh", {}).get(st.session_state.thread_id, 0))
with chat:
    st.title("Streaming explorer test")
    for message in st.session_state.messages:
        st.write(message.content)
    event = st.session_state.pop("filesystem_mention", None)
    if event and event["thread_id"] == st.session_state.thread_id:
        st.session_state.chat_prompt = f"Please look at `{event['path']}`: "
        st.session_state.mentions = st.session_state.get("mentions", 0) + 1
    st.caption(f"Composer insertions: {st.session_state.get('mentions', 0)}")
    if prompt := st.chat_input("Message", key="chat_prompt"):
        stream_and_display_response(prompt, st.empty())
