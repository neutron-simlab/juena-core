"""The Streamlit side of the server-sent-event contract.

Split out of ``juena-chatbot/app/chat_interface.py`` (00-BOUNDARY.md,
decision 7): **the SSE contract is core's; the page is the app's.** What moved
is the stream loop, the chunk classifiers, and the cards that answer an
interrupt. ``render_chat_interface()``, ``render_starter_topics()`` and the
composer stay in each application, because what a page offers before the
first message is that product's question.

Three couplings the source had to one application, each turned into a
declared parameter rather than carried across (the same treatment CP4 gave
``StreamPolicy`` and ``closing_note`` on the server):

``custom_status``
    The source's classifiers named ``sandbox_status`` directly. An
    application that emits its own status event passes a renderer for it;
    core knows only its own ``status``.

``approval``
    ``"execute_approval"`` is juena-chatbot's registered interrupt kind, not
    core's, and so are "Allow this command?" and "Running in the sandbox…".
    :class:`ApprovalCard` carries the kind name and the wording; passing
    ``None`` means this application registers no approval kind, which is v2's
    case. Core still owns the *question* card outright, because ``ask_user``
    is core's.

``session state``
    Read and written under the names in :data:`SESSION_KEYS`. Both
    applications already use these exact names, so this is shared vocabulary
    rather than one app's convention imposed on the other — but it is a
    contract either could break silently, so it is written down here and
    asserted in ``tests/test_ui_contracts.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from juena_core.clients.base import AgentClientError
from juena_core.schema.interrupts import CLARIFICATION_KIND
from juena_core.schema.server import ChatMessage
from juena_core.ui.components import finalize_streaming_message, render_streaming_token

__all__ = [
    "SESSION_KEYS",
    "ApprovalCard",
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

#: Every ``st.session_state`` name this module touches. The page owns the first
#: five and merely shares them; the last two are this module's own bookkeeping.
SESSION_KEYS = (
    "client",
    "thread_id",
    "messages",
    "selected_model",
    "selected_provider",
    "pending_approvals",
    "approval_checked_threads",
)

StatusRenderer = Callable[[Any, dict[str, Any]], None]


def _default_limits_caption(limits: Mapping[str, Any]) -> str:
    """Describe what a pending action is allowed to do.

    Core cannot phrase this: the limits are whatever the application's server
    put in the card, so it prints them rather than claiming any of them.
    """
    return " · ".join(
        f"{key.replace('_', ' ')}: {value}" for key, value in limits.items() if value
    )


@dataclass(frozen=True, slots=True)
class ApprovalCard:
    """How to render and resume one application's approval interrupt.

    ``kind`` has no default on purpose. It is the discriminator the
    application registered with
    :func:`~juena_core.server.interrupts.register_interrupt_kind` and composed
    into ``create_app(resume_input=...)``; core inventing a name here would
    send a resume body the server's union rejects.
    """

    kind: str
    heading: str = "Allow this action?"
    #: Caption shown on the spent card, per decision, so a deactivated card
    #: still says why it went quiet.
    decision_status: Mapping[str, str] = field(
        default_factory=lambda: {
            "approve": "Running…",
            "edit": "Running the edited action…",
            "reject": "Cancelled.",
        }
    )
    format_limits: Callable[[Mapping[str, Any]], str] = _default_limits_caption


def _session_list(key: str) -> list[Any]:
    value = st.session_state.get(key)
    return value if isinstance(value, list) else []


def _replace_message_by_id(messages: list[Any], chunk: ChatMessage) -> bool:
    """Overwrite an already-held message carrying the same id as *chunk*.

    Returns whether a slot was found. Messages without an id cannot be
    correlated, so they always count as new.
    """
    message_id = getattr(chunk, "id", None)
    if not message_id:
        return False
    for index, existing in enumerate(messages):
        if isinstance(existing, ChatMessage) and existing.id == message_id:
            messages[index] = chunk
            return True
    return False


def process_stream_chunk(
    chunk: Any,
    client: Any,
    response_text: str,
    received_complete_message: bool,
    message_placeholder: Any,
    messages: list[Any],
    pending_artifacts: list[dict[str, Any]] | None = None,
) -> tuple[str, bool]:
    """Fold one content chunk into the transcript and the placeholder.

    Returns the updated ``(response_text, received_complete_message)``.
    """
    if isinstance(chunk, ChatMessage):
        content_str = str(chunk.content) if chunk.content is not None else ""
        if content_str.strip().lower() == "start":
            return response_text, received_complete_message

        # Only a complete assistant message supersedes the incremental answer
        # tokens. A tool payload may be streamed before the assistant starts;
        # treating that as the completed answer makes every later token vanish
        # from the live placeholder when STREAM_TOOL_PAYLOADS is enabled.
        if chunk.type == "ai":
            received_complete_message = True
        if chunk.type == "ai" and pending_artifacts:
            stored = chunk.custom_data.get("artifacts") or []
            stored_by_id = {
                item.get("artifact_id"): item
                for item in stored
                if isinstance(item, dict) and item.get("artifact_id")
            }
            live_by_id = {
                item.get("artifact_id"): item
                for item in pending_artifacts
                if isinstance(item, dict) and item.get("artifact_id")
            }
            # Stored metadata is authoritative and keeps its order, while a
            # live event may carry content bytes or a newly-created artifact
            # that was not attached to the final message. Keep the union.
            artifact_ids = [*stored_by_id, *(key for key in live_by_id if key not in stored_by_id)]
            chunk.custom_data["artifacts"] = [
                {**stored_by_id.get(artifact_id, {}), **live_by_id.get(artifact_id, {})}
                for artifact_id in artifact_ids
            ]
            pending_artifacts.clear()
        # Resuming an interrupt replays the node that was interrupted, so the
        # stream can re-send a message this session already holds. Correlate on
        # the id the server copies from LangChain and overwrite in place, or the
        # transcript grows a second copy and its artifacts are drawn twice in
        # one script run -- which Streamlit rejects as a duplicate widget key.
        if not _replace_message_by_id(messages, chunk):
            messages.append(chunk)

        if chunk.type == "ai":
            if chunk.custom_data.get("artifacts"):
                finalize_streaming_message(
                    message_placeholder,
                    chunk.content,
                    chunk.custom_data,
                )
            else:
                finalize_streaming_message(message_placeholder, chunk.content)
            response_text = chunk.content

        return response_text, received_complete_message

    if client.is_token_message(chunk):
        if not received_complete_message:
            token_content = client.get_token_content(chunk) or ""
            if token_content:
                response_text += token_content
                render_streaming_token(response_text, message_placeholder)
        return response_text, received_complete_message

    return response_text, received_complete_message


def is_status_chunk(chunk: Any, extra_types: Iterable[str] = ()) -> bool:
    """Whether a chunk reports agent activity rather than content.

    ``extra_types`` names an application's own status events, e.g.
    ``{"sandbox_status"}``.
    """
    return isinstance(chunk, dict) and chunk.get("type") in {"status", *extra_types}


def is_approval_chunk(chunk: Any) -> bool:
    """Whether a chunk asks the user to confirm an action before it runs."""
    return isinstance(chunk, dict) and chunk.get("type") == "approval_required"


def is_interrupt_chunk(chunk: Any) -> bool:
    """Whether a chunk pauses the turn to wait on the user."""
    return isinstance(chunk, dict) and chunk.get("type") in {
        "approval_required",
        "clarification_required",
    }


def is_artifact_chunk(chunk: Any) -> bool:
    """Whether a chunk carries one generated file."""
    return isinstance(chunk, dict) and chunk.get("type") == "artifact"


def apply_status_chunk(
    status: Any,
    chunk: dict[str, Any],
    custom_status: Mapping[str, StatusRenderer] | None = None,
) -> None:
    """Reflect an activity event in the status container.

    Tool calls are written as lines inside the container so the transcript
    stays free of raw tool payloads. An application's own status event is
    handed to the renderer it registered in ``custom_status``.
    """
    chunk_type = chunk.get("type", "")
    renderer = (custom_status or {}).get(chunk_type)
    if renderer is not None:
        renderer(status, chunk)
        return

    phase = chunk.get("phase", "")
    label = chunk.get("label", "")

    if phase == "tool":
        tool = chunk.get("tool") or "tool"
        summary = chunk.get("summary") or ""
        status.write(f"🔧 `{tool}` — {summary}" if summary else f"🔧 `{tool}`")
        return

    if phase == "done":
        # First answer token: collapse the activity log and get out of the way.
        status.update(label="Done", state="complete", expanded=False)
        return

    if label:
        status.update(label=label)


def _stream_and_display_chunks(
    chunks: Iterable[Any],
    message_placeholder: Any,
    should_rerun: bool = True,
    custom_status: Mapping[str, StatusRenderer] | None = None,
) -> None:
    """Consume new-turn or resume chunks through one UI path."""
    custom_status = custom_status or {}
    initial_message_count = len(st.session_state.messages)
    response_text = ""
    received_complete_message = False
    pending_artifacts: list[dict[str, Any]] = []
    waiting_for_user = False

    try:
        with st.status("Thinking…", expanded=False) as status:
            for chunk in chunks:
                if is_status_chunk(chunk, custom_status):
                    apply_status_chunk(status, chunk, custom_status)
                    continue
                if is_artifact_chunk(chunk):
                    pending_artifacts.append(chunk)
                    continue
                if is_interrupt_chunk(chunk):
                    approvals = dict(st.session_state.get("pending_approvals") or {})
                    approvals[st.session_state.thread_id] = chunk
                    st.session_state.pending_approvals = approvals
                    waiting_for_user = True
                    status.update(
                        label=(
                            "Waiting for confirmation"
                            if is_approval_chunk(chunk)
                            else "Waiting for your answer"
                        ),
                        expanded=False,
                    )
                    continue

                response_text, received_complete_message = process_stream_chunk(
                    chunk,
                    st.session_state.client,
                    response_text,
                    received_complete_message,
                    message_placeholder,
                    st.session_state.messages,
                    pending_artifacts,
                )

            if not waiting_for_user:
                status.update(label="Done", state="complete", expanded=False)

        if response_text and not received_complete_message:
            custom_data = {"artifacts": pending_artifacts} if pending_artifacts else {}
            if custom_data:
                finalize_streaming_message(message_placeholder, response_text, custom_data)
            else:
                finalize_streaming_message(message_placeholder, response_text)
            st.session_state.messages.append(
                ChatMessage(type="ai", content=response_text, custom_data=custom_data)
            )
        elif (
            not response_text
            # A complete message has already been drawn into the placeholder by
            # process_stream_chunk. Drawing it again in the same run re-registers
            # its artifact download buttons under keys Streamlit still holds.
            and not received_complete_message
            and len(st.session_state.messages) > initial_message_count
        ):
            last_msg = st.session_state.messages[-1]
            if isinstance(last_msg, ChatMessage) and last_msg.type == "ai":
                if last_msg.custom_data.get("artifacts"):
                    finalize_streaming_message(
                        message_placeholder,
                        last_msg.content,
                        last_msg.custom_data,
                    )
                else:
                    finalize_streaming_message(message_placeholder, last_msg.content)

        if should_rerun:
            st.rerun()

    except AgentClientError as exc:
        st.error(f"Error communicating with server: {exc}")
        st.session_state.messages.append(ChatMessage(type="ai", content=f"Error: {exc}"))
    except Exception as exc:  # noqa: BLE001 - a page must not die on one bad turn
        st.error(f"Unexpected error: {exc}")
        st.session_state.messages.append(
            ChatMessage(type="ai", content=f"Unexpected error: {exc}")
        )


def stream_and_display_response(
    message: str,
    message_placeholder: Any,
    should_rerun: bool = True,
    attachments: list[Any] | None = None,
    *,
    custom_status: Mapping[str, StatusRenderer] | None = None,
) -> None:
    """Start a user turn and stream its activity and final answer."""
    chunks = st.session_state.client.stream(
        message=message,
        thread_id=st.session_state.thread_id,
        provider=st.session_state.selected_provider,
        model=st.session_state.selected_model,
        attachments=attachments,
    )
    _stream_and_display_chunks(chunks, message_placeholder, should_rerun, custom_status)


def stream_and_display_resume(
    interrupt: dict[str, Any],
    message_placeholder: Any,
    *,
    kind: str,
    custom_status: Mapping[str, StatusRenderer] | None = None,
    **fields: Any,
) -> None:
    """Answer a pending interrupt and stream the rest of the turn.

    ``kind`` and ``fields`` go straight to
    :meth:`~juena_core.clients.base.BaseAgentClient.resume_stream`, so an
    application resumes its own registered interrupt without core naming it.
    """
    thread_id = st.session_state.thread_id
    interrupt_id = str(interrupt.get("interrupt_id") or "")
    approvals = dict(st.session_state.get("pending_approvals") or {})
    approvals.pop(thread_id, None)
    st.session_state.pending_approvals = approvals
    checked_threads = set(st.session_state.get("approval_checked_threads") or ())
    checked_threads.discard(thread_id)
    st.session_state.approval_checked_threads = checked_threads

    def resume_chunks():
        try:
            yield from st.session_state.client.resume_stream(
                thread_id=thread_id,
                interrupt_id=interrupt_id,
                kind=kind,
                provider=st.session_state.selected_provider,
                model=st.session_state.selected_model,
                **fields,
            )
        except Exception:
            # The interrupt is pending again, so its card has to accept a click
            # again on the next run.
            current = dict(st.session_state.get("pending_approvals") or {})
            current[thread_id] = interrupt
            st.session_state.pending_approvals = current
            st.session_state.pop(_decision_key(thread_id, interrupt_id), None)
            st.session_state.pop(_answer_key(thread_id, interrupt_id), None)
            raise

    _stream_and_display_chunks(resume_chunks(), message_placeholder, True, custom_status)


def _decision_key(thread_id: str, interrupt_id: str) -> str:
    """Session-state key holding the decision already taken on an interrupt."""
    return f"interrupt_decision:{thread_id}:{interrupt_id}"


def _answer_key(thread_id: str, interrupt_id: str) -> str:
    """Session-state key holding a clarification answer being submitted."""

    return f"clarify_answer:{thread_id}:{interrupt_id}"


def _render_approval_card(
    approval: dict[str, Any],
    command: str,
    card: ApprovalCard,
    *,
    allowed: set[str],
    keys: str,
    disabled: bool,
    show_editor: bool,
    status: str = "",
) -> tuple[str | None, str | None, bool]:
    """Draw the confirmation card once.

    Returns ``(decision, edited_command, edit_clicked)``. With ``disabled`` set
    every control is greyed out, which lets the caller redraw the card as an
    inert copy of itself the moment a decision is taken.
    """
    st.subheader(card.heading)
    description = str(approval.get("description") or "").strip()
    if description:
        st.write(description)
    st.code(command, language="bash")
    if caption := card.format_limits(approval.get("limits") or {}):
        st.caption(caption)

    run_col, edit_col, cancel_col = st.columns(3)
    run_clicked = run_col.button(
        "Run",
        icon=":material/play_arrow:",
        type="primary",
        key=f"approval_run:{keys}",
        use_container_width=True,
        disabled=disabled or "approve" not in allowed,
    )
    edit_clicked = edit_col.button(
        "Edit",
        icon=":material/edit:",
        key=f"approval_edit_button:{keys}",
        use_container_width=True,
        disabled=disabled or "edit" not in allowed,
    )
    cancel_clicked = cancel_col.button(
        "Cancel",
        icon=":material/close:",
        key=f"approval_cancel:{keys}",
        use_container_width=True,
        disabled=disabled or "reject" not in allowed,
    )

    edited_command: str | None = None
    edit_confirmed = False
    if show_editor or edit_clicked:
        edited_command = st.text_area(
            "Command",
            value=command,
            key=f"approval_command:{keys}",
            height=140,
            disabled=disabled,
        )
        edit_confirmed = st.button(
            "Run edited command",
            icon=":material/play_arrow:",
            type="primary",
            key=f"approval_edit_confirm:{keys}",
            disabled=disabled,
        )

    if status:
        st.caption(status)

    decision = "approve" if run_clicked else "reject" if cancel_clicked else None
    if edit_confirmed:
        decision = "edit"
    return decision, edited_command, edit_clicked


def _render_question_card(
    clarification: dict[str, Any],
    *,
    keys: str,
    disabled: bool,
    status: str = "",
) -> str | None:
    """Draw the question card once, returning the user's answer when given.

    Mirrors :func:`_render_approval_card`: with ``disabled`` set every control
    is greyed out so the caller can redraw an inert copy while the turn
    resumes.
    """
    asked_by = str(clarification.get("asked_by") or "").strip()
    st.subheader("A question for you")
    if asked_by:
        st.caption(f"asked by `{asked_by}`")
    st.write(str(clarification.get("question") or ""))

    options = [
        str(option) for option in (clarification.get("options") or []) if str(option).strip()
    ]
    chosen: str | None = None
    if options:
        for index, (column, option) in enumerate(zip(st.columns(len(options)), options)):
            if column.button(
                option,
                key=f"clarify_option:{keys}:{index}",
                use_container_width=True,
                disabled=disabled,
            ):
                chosen = option

    typed = st.text_area(
        "Other…" if options else "Your answer",
        key=f"clarify_text:{keys}",
        height=90,
        disabled=disabled,
        placeholder="Answer in your own words",
    )
    sent = st.button(
        "Send",
        icon=":material/send:",
        type="primary",
        key=f"clarify_send:{keys}",
        disabled=disabled,
    )
    if status:
        st.caption(status)

    if chosen:
        return chosen
    return typed.strip() if sent and typed and typed.strip() else None


def _render_pending_clarification(
    clarification: dict[str, Any],
    custom_status: Mapping[str, StatusRenderer] | None,
) -> bool:
    """Render the question card and resume the turn once it is answered."""
    interrupt_id = str(clarification.get("interrupt_id") or "")
    if not interrupt_id:
        return False
    answer_key = _answer_key(st.session_state.thread_id, interrupt_id)
    answered = str(st.session_state.get(answer_key) or "")

    slot = st.empty()
    with slot.container(border=True):
        answer = _render_question_card(
            clarification,
            keys=interrupt_id,
            disabled=bool(answered),
            status="Answered. Continuing…" if answered else "",
        )
    if not answer:
        return False

    st.session_state[answer_key] = answer
    with slot.container(border=True):
        _render_question_card(
            clarification,
            keys=f"{interrupt_id}:answered",
            disabled=True,
            status=f"Answered: {answer}",
        )

    with st.chat_message("assistant"):
        stream_and_display_resume(
            clarification,
            st.empty(),
            kind=CLARIFICATION_KIND,
            custom_status=custom_status,
            answer=answer,
        )
    return True


def _render_pending_approval(
    approval: dict[str, Any],
    card: ApprovalCard,
    custom_status: Mapping[str, StatusRenderer] | None,
) -> bool:
    """Render the confirmation card and resume the turn once decided."""
    interrupt_id = str(approval.get("interrupt_id") or "")
    if not interrupt_id:
        return False
    command = str(approval.get("command") or "")
    allowed = set(approval.get("allowed_decisions") or ("approve", "edit", "reject"))
    edit_key = f"approval_editing:{st.session_state.thread_id}:{interrupt_id}"
    decision_key = _decision_key(st.session_state.thread_id, interrupt_id)
    decided = str(st.session_state.get(decision_key) or "")

    # The card lives in a placeholder so a decision can redraw it in place as a
    # dead control: Streamlit keeps the widgets on screen while the resumed turn
    # streams, and a second click would abort that stream mid-flight.
    slot = st.empty()
    with slot.container(border=True):
        decision, edited_command, edit_clicked = _render_approval_card(
            approval,
            command,
            card,
            allowed=allowed,
            keys=interrupt_id,
            disabled=bool(decided),
            show_editor=bool(st.session_state.get(edit_key, False)),
            status=card.decision_status.get(decided, ""),
        )

    if edit_clicked:
        st.session_state[edit_key] = True
    if decision is None:
        return False

    st.session_state[decision_key] = decision
    decided_command = edited_command if decision == "edit" and edited_command else command
    with slot.container(border=True):
        _render_approval_card(
            approval,
            decided_command,
            card,
            allowed=allowed,
            keys=f"{interrupt_id}:decided",
            disabled=True,
            show_editor=False,
            status=card.decision_status.get(decision, ""),
        )

    with st.chat_message("assistant"):
        stream_and_display_resume(
            approval,
            st.empty(),
            kind=card.kind,
            custom_status=custom_status,
            decision=decision,
            edited_command=edited_command if decision == "edit" else None,
        )
    return True


def render_pending_interrupt(
    *,
    approval: ApprovalCard | None = None,
    custom_status: Mapping[str, StatusRenderer] | None = None,
) -> bool:
    """Render controls for the current thread's pending question or approval.

    Returns whether a card was drawn, so the page can skip its composer.

    ``approval`` is ``None`` for an application that registers no approval
    kind: a pending approval then goes unrendered, which is the honest
    outcome — core has no name to resume it with. The *question* card needs no
    such argument, because ``ask_user`` is core's own.
    """
    approvals = st.session_state.get("pending_approvals") or {}
    pending = approvals.get(st.session_state.thread_id)
    checked_threads = set(st.session_state.get("approval_checked_threads") or ())
    if (
        pending is None
        and _session_list("messages")
        and st.session_state.thread_id not in checked_threads
    ):
        try:
            pending = st.session_state.client.get_pending_interrupt(
                st.session_state.thread_id,
                provider=st.session_state.selected_provider,
                model=st.session_state.selected_model,
            )
        except AgentClientError as exc:
            st.warning(f"Unable to restore what this chat was waiting for: {exc}")
        else:
            # Cache a successful empty lookup, but let the next script run
            # retry a transient transport/server failure.
            checked_threads.add(st.session_state.thread_id)
            st.session_state.approval_checked_threads = checked_threads
            if isinstance(pending, dict):
                approvals = dict(approvals)
                approvals[st.session_state.thread_id] = pending
                st.session_state.pending_approvals = approvals

    if not isinstance(pending, dict):
        return False
    if pending.get("type") == "clarification_required":
        return _render_pending_clarification(pending, custom_status)
    if pending.get("type") == "approval_required" and approval is not None:
        return _render_pending_approval(pending, approval, custom_status)
    return False
