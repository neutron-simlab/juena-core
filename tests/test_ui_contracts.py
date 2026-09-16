"""Contracts for the reusable Streamlit shell extracted in CP5."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from juena_core.clients.base import AgentClientError, BaseAgentClient
from juena_core.schema.server import ChatMessage
from juena_core.ui import client_setup, components, streaming
from juena_core.ui.chat_storage import Chat, ChatStorage, get_chat_storage


class _SessionState(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


def test_session_state_contract_is_explicit() -> None:
    assert streaming.SESSION_KEYS == (
        "client",
        "thread_id",
        "messages",
        "selected_model",
        "selected_provider",
        "pending_approvals",
        "approval_checked_threads",
    )


def test_approval_kind_has_no_application_specific_default() -> None:
    with pytest.raises(TypeError):
        streaming.ApprovalCard()  # type: ignore[call-arg]


def test_initialize_client_requires_app_identity_and_timeout() -> None:
    with pytest.raises(TypeError):
        client_setup.initialize_client("http://api.invalid")  # type: ignore[call-arg]

    client = client_setup.initialize_client(
        "http://api.invalid",
        "simulator",
        "session",
        timeout=19,
    )
    try:
        assert isinstance(client, BaseAgentClient)
        assert client.agent == "simulator"
        assert client.timeout == 19
        assert client.session_token == "session"
    finally:
        client.close()


def test_initialize_client_builds_an_application_client_subclass() -> None:
    class ApplicationClient(BaseAgentClient):
        pass

    client = client_setup.initialize_client(
        "http://api.invalid",
        "simulator",
        timeout=19,
        client_class=ApplicationClient,
    )
    try:
        assert isinstance(client, ApplicationClient)
    finally:
        client.close()


def test_chat_storage_preserves_and_filters_agent_identity() -> None:
    client = Mock(spec=BaseAgentClient)
    client.get_chat.return_value = None
    storage = ChatStorage(client)
    chat = Chat(thread_id="thread-1", agent_id="advanced_mode", title="Fit")

    storage.upsert_chat(chat)

    client.create_chat.assert_called_once_with(
        "thread-1",
        agent_id="advanced_mode",
        title="Fit",
    )

    client.list_chats.return_value = [
        {
            "thread_id": "thread-2",
            "agent_id": "simulator",
            "title": "Run",
        }
    ]
    listed = storage.list_chats(agent_id="simulator")
    assert listed[0].agent_id == "simulator"
    client.list_chats.assert_called_once_with(50, agent_id="simulator")


def test_chat_storage_is_never_shared_between_sessions() -> None:
    alice = Mock(spec=BaseAgentClient)
    bob = Mock(spec=BaseAgentClient)
    assert get_chat_storage(alice) is not get_chat_storage(bob)


def test_custom_status_events_are_dispatched_without_core_naming_them() -> None:
    status = Mock()
    renderer = Mock()
    chunk = {"type": "simulation_progress", "percent": 50}

    assert streaming.is_status_chunk(chunk, {"simulation_progress"}) is True
    streaming.apply_status_chunk(
        status,
        chunk,
        {"simulation_progress": renderer},
    )

    renderer.assert_called_once_with(status, chunk)
    status.update.assert_not_called()


def test_tool_message_does_not_suppress_later_answer_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    render_token = Mock()
    monkeypatch.setattr(streaming, "render_streaming_token", render_token)
    messages: list[ChatMessage] = []
    client = Mock()
    client.is_token_message.side_effect = (
        lambda chunk: isinstance(chunk, dict) and chunk.get("type") == "token"
    )
    client.get_token_content.side_effect = lambda chunk: chunk.get("content")

    response, complete = streaming.process_stream_chunk(
        ChatMessage(type="tool", content="tool payload"),
        client,
        "",
        False,
        Mock(),
        messages,
    )
    response, complete = streaming.process_stream_chunk(
        {"type": "token", "content": "answer"},
        client,
        response,
        complete,
        Mock(),
        messages,
    )

    assert response == "answer"
    assert complete is False
    render_token.assert_called_once()


def test_artifact_merge_keeps_stored_order_and_live_only_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streaming, "finalize_streaming_message", Mock())
    pending = [
        {"artifact_id": "stored", "content_base64": "bytes"},
        {"artifact_id": "live-only", "filename": "trace.txt"},
    ]
    message = ChatMessage(
        type="ai",
        content="done",
        custom_data={
            "artifacts": [
                {"artifact_id": "stored", "filename": "plot.png"},
            ]
        },
    )

    streaming.process_stream_chunk(
        message,
        Mock(),
        "",
        False,
        Mock(),
        [],
        pending,
    )

    assert message.custom_data["artifacts"] == [
        {
            "artifact_id": "stored",
            "filename": "plot.png",
            "content_base64": "bytes",
        },
        {"artifact_id": "live-only", "filename": "trace.txt"},
    ]
    assert pending == []


def test_failed_pending_lookup_is_retried_on_the_next_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.get_pending_interrupt.side_effect = AgentClientError("unavailable")
    state = _SessionState(
        client=client,
        thread_id="thread-1",
        messages=[ChatMessage(type="human", content="hello")],
        selected_provider="openai",
        selected_model="gpt-5-mini",
        pending_approvals={},
        approval_checked_threads=set(),
    )
    warning = Mock()
    monkeypatch.setattr(
        streaming,
        "st",
        SimpleNamespace(session_state=state, warning=warning),
    )

    assert streaming.render_pending_interrupt() is False
    assert streaming.render_pending_interrupt() is False

    assert client.get_pending_interrupt.call_count == 2
    assert state.approval_checked_threads == set()
    assert warning.call_count == 2


def test_failed_clarification_resume_reopens_its_answer_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupt = {
        "type": "clarification_required",
        "interrupt_id": "interrupt-1",
        "question": "Which model?",
    }
    client = Mock()
    client.resume_stream.side_effect = AgentClientError("unavailable")
    state = _SessionState(
        client=client,
        thread_id="thread-1",
        messages=[],
        selected_provider="openai",
        selected_model="gpt-5-mini",
        pending_approvals={"thread-1": interrupt},
        approval_checked_threads=set(),
    )
    answer_key = "clarify_answer:thread-1:interrupt-1"
    state[answer_key] = "model A"
    monkeypatch.setattr(streaming, "st", SimpleNamespace(session_state=state))

    def drain(chunks, *_args, **_kwargs) -> None:
        with pytest.raises(AgentClientError):
            list(chunks)

    monkeypatch.setattr(streaming, "_stream_and_display_chunks", drain)

    streaming.stream_and_display_resume(
        interrupt,
        Mock(),
        kind="clarification",
        answer="model A",
    )

    assert state.pending_approvals == {"thread-1": interrupt}
    assert answer_key not in state


def test_sanitize_assistant_content_removes_unrenderable_local_images() -> None:
    content = (
        "Plot:\n\n![fit](sandbox:/workspace/fit.png)\n\n"
        "![real](https://example.invalid/fit.png)"
    )
    assert components.sanitize_assistant_content(content) == (
        "Plot:\n\n![real](https://example.invalid/fit.png)"
    )


def test_artifact_history_fetch_uses_the_session_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.get_artifact.return_value = b"result\n"
    download = Mock()
    state = _SessionState(client=client)
    monkeypatch.setattr(
        components,
        "st",
        SimpleNamespace(
            session_state=state,
            caption=Mock(),
            download_button=download,
            container=lambda **_kwargs: nullcontext(),
        ),
    )

    components.render_artifacts(
        {
            "artifacts": [
                {
                    "artifact_id": "artifact-1",
                    "filename": "result.txt",
                    "kind": "file",
                    "mime_type": "text/plain",
                }
            ]
        }
    )

    client.get_artifact.assert_called_once_with("artifact-1")
    assert download.call_args.kwargs["data"] == b"result\n"
