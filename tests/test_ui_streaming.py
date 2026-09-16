"""Behaviour of the Streamlit side of the server-sent-event contract.

Moved here from juena-chatbot's ``test_chat_interface.py`` in plan 02/step 3.
Everything below drives ``juena_core.ui.streaming``: the stream loop, the chunk
classifiers, the status container and the two interrupt cards. What stayed in
juena-chatbot is the page -- starter chips, upload rules, layout -- and the
strings that describe *that* deployment's sandbox.

Three things changed in the move, all of them the couplings CP5 turned into
parameters:

* the decision a card records is keyed ``interrupt_decision:…`` rather than
  ``sandbox_decision:…``, because the kind is the application's to name;
* an approval card is drawn only when the caller supplies an
  :class:`~juena_core.ui.streaming.ApprovalCard`, so ``render_pending_interrupt``
  is given one here;
* ``stream_and_display_resume`` takes ``kind=`` and the arm's fields as
  keywords instead of a positional decision.

``test_ui_contracts.py`` holds CP5's coarser contracts for the same module,
including the clarification twin of
``test_failed_resume_reopens_the_command_for_another_click``.
"""

from types import SimpleNamespace
from unittest.mock import Mock

from juena_core.clients.base import AgentClientError
from juena_core.schema.interrupts import CLARIFICATION_KIND
from juena_core.schema.server import ChatMessage
from juena_core.ui import streaming

#: The kind juena-chatbot registers. Core names none of its own, so any test
#: that wants an approval card has to say which one it is rendering.
APPROVAL = streaming.ApprovalCard(kind="execute_approval")


class _FakeTokenClient:
    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, **kwargs):
        yield from self._chunks

    def is_token_message(self, message):
        return isinstance(message, dict) and message.get("type") == "token"

    def get_token_content(self, message):
        if self.is_token_message(message):
            return message.get("content")
        return None


class _SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStatus:
    """Stand-in for the st.status container used during streaming."""

    def __init__(self):
        self.labels: list[str] = []
        self.writes: list[str] = []
        self.state: str | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, *, label=None, state=None, expanded=None):
        if label is not None:
            self.labels.append(label)
        if state is not None:
            self.state = state

    def write(self, text):
        self.writes.append(text)


class _ButtonColumn:
    def __init__(self, clicked_label: str, calls: list[tuple[str, dict]]) -> None:
        self.clicked_label = clicked_label
        self.calls = calls

    def button(self, label: str, **kwargs):
        self.calls.append((label, kwargs))
        # Streamlit never reports a click for a greyed-out button.
        if kwargs.get("disabled"):
            return False
        return label == self.clicked_label


class _FakePlaceholder:
    """Stand-in for st.empty(), which can be redrawn into more than once."""

    def __init__(self) -> None:
        self.container_calls = 0

    def container(self, **_kwargs):
        self.container_calls += 1
        return _DummyContext()


# --------------------------------------------------------------------------- #
# What a chunk becomes
# --------------------------------------------------------------------------- #


def test_process_stream_chunk_ai_message_uses_finalize_renderer(monkeypatch) -> None:
    finalize_mock = Mock()
    message_placeholder = Mock()
    messages: list[ChatMessage] = []
    chunk = ChatMessage(type="ai", content="final response")

    monkeypatch.setattr(streaming, "finalize_streaming_message", finalize_mock)

    response_text, received_complete = streaming.process_stream_chunk(
        chunk=chunk,
        client=Mock(),
        response_text="",
        received_complete_message=False,
        message_placeholder=message_placeholder,
        messages=messages,
    )

    assert response_text == "final response"
    assert received_complete is True
    assert messages == [chunk]
    finalize_mock.assert_called_once_with(message_placeholder, "final response")
    message_placeholder.markdown.assert_not_called()


def test_process_stream_chunk_replaces_a_replayed_message(monkeypatch) -> None:
    """Resuming an interrupt replays messages the transcript already holds."""
    monkeypatch.setattr(streaming, "finalize_streaming_message", Mock())
    artifacts = [{"artifact_id": "plot-1", "filename": "plot.png"}]
    first = ChatMessage(
        type="ai",
        id="msg-1",
        content="here is the plot",
        custom_data={"artifacts": artifacts},
    )
    messages: list[ChatMessage] = [ChatMessage(type="human", content="plot"), first]
    replay = ChatMessage(
        type="ai",
        id="msg-1",
        content="here is the plot",
        custom_data={"artifacts": artifacts},
    )

    streaming.process_stream_chunk(
        chunk=replay,
        client=Mock(),
        response_text="",
        received_complete_message=False,
        message_placeholder=Mock(),
        messages=messages,
    )

    assert len(messages) == 2
    assert messages[1] is replay


def test_process_stream_chunk_appends_messages_without_an_id(monkeypatch) -> None:
    """Nothing to correlate on means the message has to count as new."""
    monkeypatch.setattr(streaming, "finalize_streaming_message", Mock())
    messages: list[ChatMessage] = [ChatMessage(type="ai", content="first")]

    streaming.process_stream_chunk(
        chunk=ChatMessage(type="ai", content="second"),
        client=Mock(),
        response_text="",
        received_complete_message=False,
        message_placeholder=Mock(),
        messages=messages,
    )

    assert [message.content for message in messages] == ["first", "second"]


def test_process_stream_chunk_ai_message_preserves_raw_math_for_finalize_renderer(
    monkeypatch,
) -> None:
    finalize_mock = Mock()
    message_placeholder = Mock()
    messages: list[ChatMessage] = []
    chunk = ChatMessage(type="ai", content=r"Result: \(a^2+b^2=c^2\)")

    monkeypatch.setattr(streaming, "finalize_streaming_message", finalize_mock)

    response_text, received_complete = streaming.process_stream_chunk(
        chunk=chunk,
        client=Mock(),
        response_text="",
        received_complete_message=False,
        message_placeholder=message_placeholder,
        messages=messages,
    )

    assert response_text == r"Result: \(a^2+b^2=c^2\)"
    assert received_complete is True
    assert messages == [chunk]
    finalize_mock.assert_called_once_with(
        message_placeholder, r"Result: \(a^2+b^2=c^2\)"
    )


def test_stream_and_display_response_token_only_completion_uses_finalize_renderer(
    monkeypatch,
) -> None:
    finalize_mock = Mock()
    render_streaming_token_mock = Mock()
    rerun_mock = Mock()
    message_placeholder = Mock()
    session_state = SimpleNamespace(
        client=_FakeTokenClient([{"type": "token", "content": "Hello"}]),
        thread_id="thread-1",
        user_id="user-1",
        selected_provider="openai",
        selected_model="gpt-4o-mini",
        messages=[],
    )
    fake_st = SimpleNamespace(
        session_state=session_state,
        rerun=rerun_mock,
        error=Mock(),
        status=lambda *a, **k: _FakeStatus(),
    )

    monkeypatch.setattr(streaming, "st", fake_st)
    monkeypatch.setattr(streaming, "finalize_streaming_message", finalize_mock)
    monkeypatch.setattr(streaming, "render_streaming_token", render_streaming_token_mock)

    streaming.stream_and_display_response(
        "hello",
        message_placeholder,
        should_rerun=False,
    )

    finalize_mock.assert_called_once_with(message_placeholder, "Hello")
    render_streaming_token_mock.assert_called_once_with("Hello", message_placeholder)
    message_placeholder.markdown.assert_not_called()
    rerun_mock.assert_not_called()
    assert len(session_state.messages) == 1
    assert session_state.messages[0].type == "ai"
    assert session_state.messages[0].content == "Hello"


def test_stream_and_display_response_token_only_math_completion_keeps_raw_text(
    monkeypatch,
) -> None:
    finalize_mock = Mock()
    render_streaming_token_mock = Mock()
    message_placeholder = Mock()
    session_state = SimpleNamespace(
        client=_FakeTokenClient(
            [
                {"type": "token", "content": r"Equation: \("},
                {"type": "token", "content": r"a+b\)"},
            ]
        ),
        thread_id="thread-1",
        user_id="user-1",
        selected_provider="openai",
        selected_model="gpt-4o-mini",
        messages=[],
    )
    fake_st = SimpleNamespace(
        session_state=session_state,
        rerun=Mock(),
        error=Mock(),
        status=lambda *a, **k: _FakeStatus(),
    )

    monkeypatch.setattr(streaming, "st", fake_st)
    monkeypatch.setattr(streaming, "finalize_streaming_message", finalize_mock)
    monkeypatch.setattr(streaming, "render_streaming_token", render_streaming_token_mock)

    streaming.stream_and_display_response(
        "hello",
        message_placeholder,
        should_rerun=False,
    )

    assert [call.args for call in render_streaming_token_mock.call_args_list] == [
        (r"Equation: \(", message_placeholder),
        (r"Equation: \(a+b\)", message_placeholder),
    ]
    finalize_mock.assert_called_once_with(message_placeholder, r"Equation: \(a+b\)")
    assert session_state.messages[0].content == r"Equation: \(a+b\)"


def test_status_chunks_are_recognized() -> None:
    assert streaming.is_status_chunk({"type": "status", "phase": "thinking"}) is True
    assert streaming.is_status_chunk({"type": "token", "content": "x"}) is False
    assert streaming.is_status_chunk(ChatMessage(type="ai", content="x")) is False


def test_approval_and_artifact_chunks_are_recognized() -> None:
    assert streaming.is_approval_chunk({"type": "approval_required"}) is True
    assert streaming.is_artifact_chunk({"type": "artifact"}) is True
    assert streaming.is_artifact_chunk({"type": "token"}) is False


# --------------------------------------------------------------------------- #
# The two interrupt cards
# --------------------------------------------------------------------------- #


def test_pending_approval_offers_one_click_choices(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []
    approval = {
        "type": "approval_required",
        "interrupt_id": "interrupt-1",
        "command": "python plot.py",
        "description": "Create a plot",
        "limits": {"cpu": "2 vCPU", "memory": "4 GB"},
        "allowed_decisions": ["approve", "edit", "reject"],
    }
    session_state = _SessionState(
        thread_id="thread-1",
        messages=[ChatMessage(type="human", content="plot")],
        pending_approvals={"thread-1": approval},
        selected_provider="openai",
        selected_model="gpt-4o-mini",
    )
    slot = _FakePlaceholder()
    fake_st = SimpleNamespace(
        session_state=session_state,
        container=lambda **_kwargs: _DummyContext(),
        subheader=Mock(),
        write=Mock(),
        code=Mock(),
        caption=Mock(),
        columns=lambda _count: [
            _ButtonColumn("Run", calls),
            _ButtonColumn("Run", calls),
            _ButtonColumn("Run", calls),
        ],
        chat_message=lambda _role: _DummyContext(),
        empty=Mock(side_effect=[slot, Mock()]),
    )
    resume_mock = Mock()
    monkeypatch.setattr(streaming, "st", fake_st)
    monkeypatch.setattr(streaming, "stream_and_display_resume", resume_mock)

    assert streaming.render_pending_interrupt(approval=APPROVAL) is True

    # The card is drawn twice into the same placeholder: live, then spent.
    assert slot.container_calls == 2
    assert [label for label, _kwargs in calls] == [
        "Run",
        "Edit",
        "Cancel",
        "Run",
        "Edit",
        "Cancel",
    ]
    assert [kwargs["disabled"] for _label, kwargs in calls[3:]] == [True, True, True]
    assert session_state["interrupt_decision:thread-1:interrupt-1"] == "approve"
    resume_mock.assert_called_once()
    assert resume_mock.call_args.args[0] is approval
    assert resume_mock.call_args.kwargs["kind"] == "execute_approval"
    assert resume_mock.call_args.kwargs["decision"] == "approve"


def test_pending_approval_ignores_clicks_once_a_decision_is_taken(monkeypatch) -> None:
    """A card whose decision is already in flight offers no live controls."""
    calls: list[tuple[str, dict]] = []
    approval = {
        "type": "approval_required",
        "interrupt_id": "interrupt-1",
        "command": "python plot.py",
        "allowed_decisions": ["approve", "edit", "reject"],
    }
    session_state = _SessionState(
        thread_id="thread-1",
        messages=[ChatMessage(type="human", content="plot")],
        pending_approvals={"thread-1": approval},
        selected_provider="openai",
        selected_model="gpt-4o-mini",
    )
    session_state["interrupt_decision:thread-1:interrupt-1"] = "approve"
    slot = _FakePlaceholder()
    fake_st = SimpleNamespace(
        session_state=session_state,
        container=lambda **_kwargs: _DummyContext(),
        subheader=Mock(),
        write=Mock(),
        code=Mock(),
        caption=Mock(),
        columns=lambda _count: [
            _ButtonColumn("Run", calls),
            _ButtonColumn("Run", calls),
            _ButtonColumn("Run", calls),
        ],
        chat_message=lambda _role: _DummyContext(),
        empty=Mock(return_value=slot),
    )
    resume_mock = Mock()
    monkeypatch.setattr(streaming, "st", fake_st)
    monkeypatch.setattr(streaming, "stream_and_display_resume", resume_mock)

    assert streaming.render_pending_interrupt(approval=APPROVAL) is False

    assert all(kwargs["disabled"] for _label, kwargs in calls)
    resume_mock.assert_not_called()


def test_an_approval_goes_unrendered_when_no_card_is_supplied(monkeypatch) -> None:
    """The honest outcome for an application that registered no approval kind.

    Core has no name to resume this interrupt with, so it draws nothing rather
    than a card whose buttons would post a body the server's union rejects.
    Vitess v2 is exactly this case.
    """
    session_state = _SessionState(
        thread_id="thread-1",
        messages=[ChatMessage(type="human", content="plot")],
        pending_approvals={
            "thread-1": {
                "type": "approval_required",
                "interrupt_id": "interrupt-1",
                "command": "python plot.py",
            }
        },
        selected_provider="openai",
        selected_model="gpt-4o-mini",
    )
    calls: list[tuple[str, dict]] = []
    subheader = Mock()
    # Everything a card would need, so that nothing but the missing
    # ``ApprovalCard`` can be the reason no card appears.
    monkeypatch.setattr(
        streaming,
        "st",
        SimpleNamespace(
            session_state=session_state,
            container=lambda **_kwargs: _DummyContext(),
            subheader=subheader,
            write=Mock(),
            code=Mock(),
            caption=Mock(),
            columns=lambda count: [_ButtonColumn("Run", calls) for _ in range(count)],
            chat_message=lambda _role: _DummyContext(),
            empty=Mock(return_value=_FakePlaceholder()),
        ),
    )

    assert streaming.render_pending_interrupt() is False
    subheader.assert_not_called()
    assert calls == []


def test_pending_question_offers_suggestions_and_a_free_text_answer(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []
    clarification = {
        "type": "clarification_required",
        "interrupt_id": "interrupt-2",
        "asked_by": "software-specialist",
        "question": "Which background subtraction?",
        "options": ["Solvent-only", "Empty cell"],
    }
    session_state = _SessionState(
        thread_id="thread-1",
        messages=[ChatMessage(type="human", content="reduce this")],
        pending_approvals={"thread-1": clarification},
        selected_provider="openai",
        selected_model="gpt-4o-mini",
    )
    slot = _FakePlaceholder()
    fake_st = SimpleNamespace(
        session_state=session_state,
        container=lambda **_kwargs: _DummyContext(),
        subheader=Mock(),
        write=Mock(),
        caption=Mock(),
        text_area=Mock(return_value=""),
        button=Mock(return_value=False),
        columns=lambda count: [_ButtonColumn("Empty cell", calls) for _ in range(count)],
        chat_message=lambda _role: _DummyContext(),
        empty=Mock(side_effect=[slot, Mock()]),
    )
    resume_mock = Mock()
    monkeypatch.setattr(streaming, "st", fake_st)
    monkeypatch.setattr(streaming, "stream_and_display_resume", resume_mock)

    # No ApprovalCard: the question card is core's own, because ask_user is.
    assert streaming.render_pending_interrupt() is True

    # Live card, then a spent copy redrawn into the same placeholder.
    assert slot.container_calls == 2
    assert [label for label, _kwargs in calls[:2]] == ["Solvent-only", "Empty cell"]
    assert all(kwargs["disabled"] for _label, kwargs in calls[2:])
    assert session_state["clarify_answer:thread-1:interrupt-2"] == "Empty cell"
    resume_mock.assert_called_once()
    assert resume_mock.call_args.kwargs["answer"] == "Empty cell"
    assert resume_mock.call_args.kwargs["kind"] == CLARIFICATION_KIND
    assert "decision" not in resume_mock.call_args.kwargs


def test_a_question_waits_until_an_answer_is_actually_given(monkeypatch) -> None:
    """Rendering the card must not resume the turn on its own."""
    calls: list[tuple[str, dict]] = []
    clarification = {
        "type": "clarification_required",
        "interrupt_id": "interrupt-2",
        "asked_by": "juena",
        "question": "Which instrument?",
        "options": [],
    }
    session_state = _SessionState(
        thread_id="thread-1",
        messages=[ChatMessage(type="human", content="explain")],
        pending_approvals={"thread-1": clarification},
        selected_provider="openai",
        selected_model="gpt-4o-mini",
    )
    fake_st = SimpleNamespace(
        session_state=session_state,
        container=lambda **_kwargs: _DummyContext(),
        subheader=Mock(),
        write=Mock(),
        caption=Mock(),
        text_area=Mock(return_value="   "),
        button=Mock(return_value=True),
        columns=lambda count: [_ButtonColumn("", calls) for _ in range(count)],
        chat_message=lambda _role: _DummyContext(),
        empty=Mock(return_value=_FakePlaceholder()),
    )
    resume_mock = Mock()
    monkeypatch.setattr(streaming, "st", fake_st)
    monkeypatch.setattr(streaming, "stream_and_display_resume", resume_mock)

    # Send was clicked, but the box holds only whitespace.
    assert streaming.render_pending_interrupt() is False
    resume_mock.assert_not_called()


def test_failed_resume_reopens_the_command_for_another_click(monkeypatch) -> None:
    """A resume that never starts must leave a clickable card behind."""
    approval = {
        "type": "approval_required",
        "interrupt_id": "interrupt-1",
        "command": "python plot.py",
    }
    client = Mock()
    client.resume_stream.side_effect = AgentClientError("boom")
    session_state = _SessionState(
        thread_id="thread-1",
        messages=[],
        pending_approvals={"thread-1": approval},
        approval_checked_threads=set(),
        client=client,
        selected_provider="openai",
        selected_model="gpt-4o-mini",
    )
    session_state["interrupt_decision:thread-1:interrupt-1"] = "approve"
    monkeypatch.setattr(
        streaming,
        "st",
        SimpleNamespace(session_state=session_state),
    )

    def drain(chunks, *_args, **_kwargs):
        """Stand in for the display loop, which swallows client errors."""
        try:
            list(chunks)
        except AgentClientError:
            pass

    monkeypatch.setattr(streaming, "_stream_and_display_chunks", drain)

    streaming.stream_and_display_resume(
        approval,
        Mock(),
        kind="execute_approval",
        decision="approve",
    )

    assert session_state["pending_approvals"] == {"thread-1": approval}
    assert "interrupt_decision:thread-1:interrupt-1" not in session_state


# --------------------------------------------------------------------------- #
# The status container
# --------------------------------------------------------------------------- #


def test_tool_activity_is_logged_inside_the_status_container() -> None:
    """Tool activity belongs in the collapsible status, not the transcript."""
    status = _FakeStatus()

    streaming.apply_status_chunk(
        status,
        {
            "type": "status",
            "phase": "tool",
            "label": "Ran search_code_semantic",
            "tool": "search_code_semantic",
            "summary": "12 hits",
        },
    )

    assert status.writes == ["🔧 `search_code_semantic` — 12 hits"]
    assert status.labels == []


def test_status_relabels_and_completes() -> None:
    status = _FakeStatus()

    streaming.apply_status_chunk(
        status, {"type": "status", "phase": "researching", "label": "Researching…"}
    )
    assert status.labels == ["Researching…"]
    assert status.state is None

    # The first answer token collapses the activity log.
    streaming.apply_status_chunk(status, {"type": "status", "phase": "done", "label": ""})
    assert status.state == "complete"


def test_stream_routes_status_chunks_away_from_response_text(monkeypatch) -> None:
    """A status event must never be accumulated into the answer."""
    status = _FakeStatus()
    finalize_mock = Mock()
    session_state = SimpleNamespace(
        client=_FakeTokenClient(
            [
                {"type": "status", "phase": "researching", "label": "Researching…"},
                {"type": "status", "phase": "tool", "tool": "grep", "summary": "3 hits"},
                {"type": "token", "content": "Answer"},
            ]
        ),
        thread_id="thread-1",
        user_id="user-1",
        selected_provider="openai",
        selected_model="gpt-4o-mini",
        messages=[],
    )
    fake_st = SimpleNamespace(
        session_state=session_state,
        rerun=Mock(),
        error=Mock(),
        status=lambda *a, **k: status,
    )

    monkeypatch.setattr(streaming, "st", fake_st)
    monkeypatch.setattr(streaming, "finalize_streaming_message", finalize_mock)
    monkeypatch.setattr(streaming, "render_streaming_token", Mock())

    streaming.stream_and_display_response("hi", Mock(), should_rerun=False)

    # Only the token text became the answer; the status events did not.
    assert len(session_state.messages) == 1
    assert session_state.messages[0].content == "Answer"
    assert "Researching…" in status.labels
    assert status.writes == ["🔧 `grep` — 3 hits"]
