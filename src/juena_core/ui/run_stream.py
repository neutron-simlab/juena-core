"""Keep a response reader alive across Streamlit page reruns.

The background thread owns its event loop and HTTP connection. It only fills a
locked buffer; the page drains and renders that buffer in ui.streaming. Session
release cancels the reader, which closes the response and stops the API run.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from time import monotonic

import streamlit as st

from juena_core.clients.base import AgentClientError
from juena_core.schema.server import ChatMessage

__all__ = [
    "StreamController",
    "get_stream_controller",
    "start_response",
    "start_resume",
]


@dataclass
class LiveView:
    """What the live reply has shown so far, kept across page reruns.

    Attributes:
        text: The answer text received so far.
        complete: Whether the complete answer message has arrived.
        steps: The activity log's steps.
        artifacts: Files streamed during the turn.
        started: When the turn started, from ``time.monotonic``.
        label: The status box's label.
        interrupt: The question or approval the turn paused on, if any.
        resume_interrupt: The interrupt this turn resumes, if any, so it can
            be offered again if the reply fails.
        last_message: The last complete assistant message.
    """

    text: str = ""
    complete: bool = False
    steps: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    started: float = field(default_factory=monotonic)
    label: str = "Thinking…"
    interrupt: dict | None = None
    resume_interrupt: dict | None = None
    last_message: ChatMessage | None = None


class StreamController:
    """One browser session's background response reader.

    Only plain data crosses the worker boundary; all st calls stay on-page.
    """

    def __init__(self):
        """Initialize an idle controller."""
        self.lock = threading.Lock()
        self.buffer = deque()
        self.thread: threading.Thread | None = None
        self.loop = None
        self.task = None
        self.done = True
        self.run: dict = {}
        self.thread_id: str | None = None
        self.view = LiveView()
        self.error: str | None = None
        self.closed = threading.Event()

    @property
    def active(self):
        """Whether a response is being read."""
        with self.lock:
            return not self.done

    def push(self, chunk):
        """Buffer one event, remembering the latest run snapshot.

        Args:
            chunk: One streamed event.
        """
        with self.lock:
            if isinstance(chunk, dict) and chunk.get("type") == "run":
                self.run = chunk
            self.buffer.append(chunk)

    def drain(self):
        """Take available events without blocking the page on the network.

        Returns:
            The buffered events, oldest first.
        """
        with self.lock:
            chunks = list(self.buffer)
            self.buffer.clear()
            return chunks

    def start(self, thread_id, chunks, *, interrupt=None):
        """Consume one async stream; rerenders reuse this same reader.

        Args:
            thread_id: The conversation the reply belongs to.
            chunks: The async stream of events.
            interrupt: The interrupt being resumed, if any.

        Raises:
            RuntimeError: A response is already running.
        """
        if self.active:
            raise RuntimeError("A response is already running")
        self.thread_id = thread_id
        self.run, self.error, self.view = {}, None, LiveView(resume_interrupt=interrupt)
        self.loop = self.task = None
        self.closed.clear()
        with self.lock:
            self.done = False
            self.buffer.clear()

        async def receive():
            self.loop, self.task = asyncio.get_running_loop(), asyncio.current_task()
            try:
                if self.closed.is_set():
                    return
                async for chunk in chunks:
                    self.push(chunk)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if isinstance(exc, AgentClientError):
                    self.error = str(exc)
                else:
                    self.error = "The response connection failed"
                self.push({"type": "error", "content": self.error})
            finally:
                await chunks.aclose()
                terminal = self.run.get("state") in {
                    "completed",
                    "paused",
                    "cancelled",
                    "failed",
                }
                if not self.closed.is_set() and self.error is None and not terminal:
                    self.error = (
                        "The response connection ended before completion was confirmed"
                    )
                    self.push({"type": "error", "content": self.error})
                with self.lock:
                    self.done = True

        self.thread = threading.Thread(
            target=lambda: asyncio.run(receive()), daemon=True, name="juena-response"
        )
        self.thread.start()

    def close(self):
        """Cancel HTTP without calling Streamlit from the release callback."""
        self.closed.set()
        loop, task = self.loop, self.task
        if loop is not None and task is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # The HTTP reader finished between the check and call.
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)


@st.cache_resource(scope="session", on_release=lambda controller: controller.close())
def get_stream_controller():
    """Return this browser session's controller, closed when the session ends.

    Returns:
        The controller shared by every rerun of this session.
    """
    return StreamController()


def start_response(message, *, attachments=None):
    """Start reading the reply to a user message, then rerun the page.

    Args:
        message: The user's message.
        attachments: Files sent with it, if any.
    """
    client = st.session_state.client
    controller = get_stream_controller()
    chunks = client.astream(
        message,
        st.session_state.selected_model,
        st.session_state.selected_provider,
        st.session_state.thread_id,
        attachments,
    )
    controller.start(st.session_state.thread_id, chunks)
    st.rerun()


def start_resume(interrupt, *, kind, **fields):
    """Start reading the rest of a paused turn, then rerun the page.

    The activity log the turn paused with is carried over.

    Args:
        interrupt: The pending interrupt being answered.
        kind: The interrupt kind the server resumes it as.
        **fields: The answer's fields for that kind.
    """
    client = st.session_state.client
    controller = get_stream_controller()
    thread_id = st.session_state.thread_id
    chunks = client.aresume_stream(
        thread_id=thread_id,
        interrupt_id=interrupt["interrupt_id"],
        kind=kind,
        provider=st.session_state.selected_provider,
        model=st.session_state.selected_model,
        **fields,
    )
    controller.start(thread_id, chunks, interrupt=interrupt)
    carried = dict(st.session_state.get("pending_activity") or {})
    record = carried.pop(thread_id, {})
    st.session_state.pending_activity = carried
    controller.view.steps = record.get("steps", [])
    controller.view.started -= record.get("seconds", 0)
    st.rerun()
