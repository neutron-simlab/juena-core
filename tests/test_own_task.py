"""A stopped stream must finish its cleanup before the response ends.

The responses here are real sse-starlette responses, driven at the ASGI level
and ended the way a real one ends when the client leaves: sse-starlette's
anyio cancel scope, which cancels again at every later ``await``. The source
stands in for an agent run. On the way out it *waits* for something, like
LangGraph waiting for its running nodes, and only then counts as cleaned up.

Two moments to be cancelled at:

- **waiting**: the run is mid-answer, and the response is waiting for its
  next event;
- **sending**: an event is being written to a client that stopped reading,
  so the generator feeding the response is paused at its ``yield``.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from sse_starlette import EventSourceResponse

from juena_core.server.streaming.own_task import AgentStreamResponse, run_in_own_task

MOMENTS = ["waiting", "sending"]


def _run(log: list[str], *, moment: str, cleanup_seconds: float = 0.05):
    async def events():
        try:
            number = 0
            while True:
                yield f"event {number}"
                number += 1
                # Mid-answer with nothing to send yet, or streaming steadily.
                await asyncio.sleep(10 if moment == "waiting" else 0.01)
        finally:
            await asyncio.sleep(cleanup_seconds)
            log.append("cleaned")

    return events()


async def _serve_until_the_client_leaves(response, *, moment: str) -> None:
    left = asyncio.Event()
    first_event_sent = asyncio.Event()

    async def receive():
        await left.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_event_sent.set()
            if moment == "sending":
                await asyncio.Event().wait()  # a client that stopped reading

    serving = asyncio.create_task(response({"type": "http"}, receive, send))
    await first_event_sent.wait()
    await asyncio.sleep(0.05)
    left.set()
    await asyncio.wait_for(serving, timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("moment", MOMENTS)
async def test_a_plain_response_loses_the_cleanup(moment: str) -> None:
    """The failure being fixed. Without it, the test below proves nothing."""

    log: list[str] = []
    await _serve_until_the_client_leaves(
        EventSourceResponse(_run(log, moment=moment)), moment=moment
    )

    assert log == []


@pytest.mark.asyncio
@pytest.mark.parametrize("moment", MOMENTS)
async def test_the_run_is_cleaned_up_before_the_thread_is_released(moment: str) -> None:
    log: list[str] = []
    response = AgentStreamResponse(
        _run(log, moment=moment), on_close=lambda: log.append("released")
    )

    await _serve_until_the_client_leaves(response, moment=moment)

    assert log == ["cleaned", "released"]


@pytest.mark.asyncio
async def test_items_arrive_in_order() -> None:
    async def numbers():
        for number in range(5):
            yield number

    async with run_in_own_task(numbers()) as items:
        assert [item async for item in items] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_an_error_in_the_run_reaches_the_reader() -> None:
    async def failing():
        yield "first"
        raise ValueError("model provider down")

    received = []
    with pytest.raises(ValueError, match="model provider down"):
        async with run_in_own_task(failing()) as items:
            async for item in items:
                received.append(item)
    assert received == ["first"]


@pytest.mark.asyncio
async def test_a_slow_cleanup_cannot_hold_the_response_past_its_deadline() -> None:
    log: list[str] = []
    started = time.monotonic()

    async with run_in_own_task(
        _run(log, moment="sending", cleanup_seconds=1.0), cleanup_seconds=0.1
    ) as items:
        async for _ in items:
            break

    assert time.monotonic() - started < 0.5
    # At the deadline the wait is abandoned, which cancels the run a second
    # time: its cleanup is cut short, but nothing is left running either.
    assert log == []
    assert asyncio.all_tasks() == {asyncio.current_task()}
