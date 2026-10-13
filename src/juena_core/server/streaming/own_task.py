"""A streamed agent run that is always stopped cleanly, however its response ends.

sse-starlette serves each response inside an anyio cancel scope. When the
client disconnects or the server shuts down, that scope cancels the request,
and it keeps cancelling it at every later ``await``. Two things went wrong
when the agent ran directly inside that response:

- **Cancelled while waiting for the next event.** LangGraph's cleanup cancels
  its running node tasks and then *waits* for them. That wait is one of those
  later awaits, so the cleanup was cut short.
- **Cancelled while sending an event.** The generator feeding the response
  was paused at a ``yield``, so its ``finally`` never ran at all.

Either way, a node went on running after its stream was gone. It kept calling
the model, kept the conversation reserved, and its tracing spans ended only
after the server had already flushed its tracer.

So the response owns the run instead. The agent's generator runs in an
ordinary asyncio task of its own, outside the cancel scope. Whatever ends the
response, its exit cancels that task once, waits for the cleanup in a shielded
scope (up to a deadline), and only then releases the conversation.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

import anyio
from sse_starlette import EventSourceResponse
from starlette.types import Receive, Scope, Send

__all__ = ["CLEANUP_SECONDS", "AgentStreamResponse", "run_in_own_task"]

T = TypeVar("T")

#: How long a stopped run may take to unwind before the response ends anyway.
#: Past it the run is cancelled once more. Within uvicorn's 10 s
#: graceful-shutdown wait in the chatbot's main.py.
CLEANUP_SECONDS = 5.0

_ITEM, _ERROR, _DONE = "item", "error", "done"


@asynccontextmanager
async def run_in_own_task(
    source: AsyncIterator[T], *, cleanup_seconds: float = CLEANUP_SECONDS
) -> AsyncGenerator[AsyncIterator[T], None]:
    """Run ``source`` in its own task while the context is open; yield its items.

    Leaving the context, for any reason, cancels the task and waits for it.
    An error in ``source`` is raised by the iterator, exactly as iterating
    ``source`` directly would have raised it. The queue is unbounded, so the
    run is never held back by a slow client; a chat's events are small.
    """

    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for item in source:
                queue.put_nowait((_ITEM, item))
        except BaseException as error:
            queue.put_nowait((_ERROR, error))
            raise
        queue.put_nowait((_DONE, None))

    async def items() -> AsyncGenerator[T, None]:
        while True:
            kind, value = await queue.get()
            if kind == _ERROR:
                raise value  # type: ignore[misc]
            if kind == _DONE:
                return
            yield value  # type: ignore[misc]

    task = asyncio.create_task(pump())
    events = items()
    try:
        yield events
    finally:
        task.cancel()
        with anyio.move_on_after(cleanup_seconds, shield=True):
            await asyncio.gather(task, return_exceptions=True)
            # It may be paused at a `yield` the response never came back for.
            await events.aclose()


class AgentStreamResponse(EventSourceResponse):
    """An SSE response that owns its agent run, from first event to cleanup.

    ``on_close`` runs last, after the run has been stopped and cleaned up.
    It runs however the response ended: finished, client gone, or server
    stopping.
    """

    def __init__(
        self,
        source: AsyncIterator[str],
        *,
        on_close: Callable[[], None],
        cleanup_seconds: float = CLEANUP_SECONDS,
    ) -> None:
        super().__init__(source)
        self._source = source
        self._on_close = on_close
        self._cleanup_seconds = cleanup_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            async with run_in_own_task(
                self._source, cleanup_seconds=self._cleanup_seconds
            ) as events:
                self.body_iterator = events
                await super().__call__(scope, receive, send)
        finally:
            self._on_close()
