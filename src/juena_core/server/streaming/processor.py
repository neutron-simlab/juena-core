"""Turn LangGraph stream events into the SSE events the UI consumes.

Two rules shape this module:

1. Only the supervisor's tokens are the answer. A subagent's tokens are its
   internal reasoning; streaming them makes the UI show text that is later
   replaced by the real answer.
2. Tool results are activity, not transcript. They are large, the model has
   already read them, and rendering them buries the answer. They become compact
   ``status`` events unless payloads are explicitly enabled.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage, ToolMessage

from juena_core.log import get_logger
from juena_core.server.errors import MessageProcessingError, StreamingError
from juena_core.server.interrupts import interrupt_event
from juena_core.server.streaming import events
from juena_core.server.streaming.handlers import (
    extract_interrupts,
    extract_update_messages,
    is_subagent_namespace,
    token_text,
)
from juena_core.server.utils import langchain_to_chat_message

logger = get_logger(__name__)

__all__ = ["StreamPolicy", "StreamEventProcessor"]

# Preview length for a tool result shown inside the status container.
_TOOL_SUMMARY_CHARS = 120


@dataclass(frozen=True, slots=True)
class StreamPolicy:
    """What an application's execution machinery contributes to the stream.

    Both fields name things core cannot know, and both default to empty so a
    core-only application streams correctly without declaring anything.

    ``custom_event_types`` is an allowlist, not a filter: the ``custom`` stream
    mode carries whatever any middleware or library chose to write, so
    forwarding by default would put internal payloads on a user's wire.

    ``silent_tools`` are tools whose results never appear, not even as the
    one-line activity preview. juena-chatbot lists its sandbox ``execute``,
    which has dedicated bounded status events and whose stdout may be very
    large or binary-like.
    """

    custom_event_types: frozenset[str] = field(default_factory=frozenset)
    silent_tools: frozenset[str] = field(default_factory=frozenset)


def _tool_summary(message: ToolMessage) -> str:
    """One-line gist of a tool result, for the activity log."""

    text = " ".join(message.text.split())
    if len(text) <= _TOOL_SUMMARY_CHARS:
        return text
    return text[:_TOOL_SUMMARY_CHARS].rstrip() + "…"


class StreamEventProcessor:
    """Processor for LangGraph stream events."""

    def __init__(
        self,
        agent: Any,
        config: Any,
        run_id: str,
        user_input_message: str,
        *,
        include_tool_payloads: bool = False,
        ignored_interrupt_ids: set[str] | None = None,
        policy: StreamPolicy | None = None,
    ) -> None:
        self.agent = agent
        self.config = config
        self.run_id = run_id
        self.user_input_message = user_input_message
        self.include_tool_payloads = include_tool_payloads
        self.ignored_interrupt_ids = frozenset(ignored_interrupt_ids or ())
        self.policy = policy or StreamPolicy()

        # Messages already sent this run, keyed by LangChain message id.
        # `updates` only emits new messages -- including on a resumed thread --
        # so this guards against the same message arriving twice within one run,
        # not against replayed history.
        self._streamed_message_ids: set[str] = set()
        self._streamed_interrupt_ids: set[str] = set()

        # Tracks the last status emitted so identical ones are not repeated.
        self._last_status: tuple[str, str] | None = None

    def _parse_stream_event(self, stream_event: Any) -> tuple[str, Any, Any]:
        """Split a raw stream event into (stream_mode, event, node_path)."""

        if isinstance(stream_event, dict) and {"type", "data"}.issubset(stream_event):
            return stream_event["type"], stream_event["data"], stream_event.get("ns")
        if not isinstance(stream_event, tuple):
            raise StreamingError(f"Unexpected stream event type: {type(stream_event)}")

        if len(stream_event) == 3:
            # subgraphs=True: (node_path, stream_mode, event)
            node_path, stream_mode, event = stream_event
            return stream_mode, event, node_path

        stream_mode, event = stream_event
        return stream_mode, event, None

    def _status(self, phase: str, label: str, **extra: Any) -> dict[str, Any] | None:
        """Build a status event, or None when it repeats the previous one."""

        key = (phase, label)
        if key == self._last_status:
            return None
        self._last_status = key
        return events.status_event(phase, label, **extra)

    async def process_event(self, stream_event: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Process one stream event, yielding SSE payload dicts."""

        try:
            stream_mode, event, node_path = self._parse_stream_event(stream_event)
            from_subagent = is_subagent_namespace(node_path)

            if stream_mode == "custom":
                if (
                    isinstance(event, dict)
                    and event.get("type") in self.policy.custom_event_types
                ):
                    yield event
                return

            if stream_mode == "messages":
                if from_subagent:
                    # Subagent reasoning is activity, not the answer.
                    status = self._status(events.PHASE_RESEARCHING, "Consulting a specialist…")
                    if status:
                        yield status
                    return

                text = token_text(event)
                if text:
                    status = self._status(events.PHASE_DONE, "")
                    if status:
                        yield status
                    yield events.token_event(text)
                return

            if stream_mode == "updates":
                for interrupt in extract_interrupts(event):
                    if (
                        interrupt.id in self.ignored_interrupt_ids
                        or interrupt.id in self._streamed_interrupt_ids
                    ):
                        continue
                    payload = interrupt_event(interrupt)
                    if payload is not None:
                        self._streamed_interrupt_ids.add(interrupt.id)
                        yield payload
                for message in extract_update_messages(event):
                    async for payload in self._handle_message(message, from_subagent):
                        yield payload

        except Exception as e:
            logger.error(f"Error processing stream event: {e}", exc_info=True)
            error = StreamingError("Failed to process stream event", details={"error": str(e)})
            yield events.error_event(error.message)

    async def _handle_message(
        self, message: BaseMessage, from_subagent: bool
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert one complete message into the events the client should see."""

        try:
            if isinstance(message, SystemMessage):
                return

            # Summarization deletes old turns by emitting `RemoveMessage` into
            # the same stream. It is an instruction to the state, not something
            # a person reads, and rendering it raised an error the user saw
            # instead of their answer.
            if isinstance(message, RemoveMessage):
                return

            if isinstance(message, ToolMessage):
                tool_name = getattr(message, "name", None) or "tool"
                if tool_name in self.policy.silent_tools:
                    return
                yield events.status_event(
                    events.PHASE_TOOL,
                    f"Ran {tool_name}",
                    tool=tool_name,
                    summary=_tool_summary(message),
                )
                if not self.include_tool_payloads:
                    return

            chat_message = langchain_to_chat_message(message)
            chat_message.run_id = str(self.run_id)

            # A specialist's final report is input to the supervisor, not an
            # answer for the user; the supervisor synthesizes from it.
            if from_subagent and chat_message.type == "ai":
                return

            # Echo of the user's own turn.
            if chat_message.type == "human" and chat_message.content == self.user_input_message:
                return
            if chat_message.type == "system":
                return

            # Tool-call announcements carry no content of their own.
            if (
                chat_message.type == "ai"
                and chat_message.tool_calls
                and not (chat_message.content or "").strip()
            ):
                return

            message_id = chat_message.id
            if message_id:
                if message_id in self._streamed_message_ids:
                    return
                self._streamed_message_ids.add(message_id)

            yield events.message_event(chat_message)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            error = MessageProcessingError(
                "Failed to process message",
                message_type=type(message).__name__,
                details={"error": str(e)},
            )
            yield events.error_event(error.message)
