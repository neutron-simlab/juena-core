"""The session reader stays alive across page reruns and closes on release."""

import asyncio
import threading

import httpx
from fastapi import FastAPI
from starlette.responses import StreamingResponse

from juena_core.clients.base import BaseAgentClient
from juena_core.ui.run_stream import StreamController


def test_reader_receives_without_streamlit_and_close_cancels_a_blocked_connection(
    monkeypatch,
):
    entered, cancelled = threading.Event(), threading.Event()
    app = FastAPI()

    @app.post("/juena/stream")
    async def stream():
        async def source():
            entered.set()
            try:
                await asyncio.Event().wait()
                yield ""
            finally:
                cancelled.set()

        return StreamingResponse(source(), media_type="text/event-stream")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda: real_client(transport=httpx.ASGITransport(app=app)),
    )
    controller = StreamController()
    client = BaseAgentClient("http://fixture", agent="juena")
    controller.start("thread", client.astream("original"))
    try:
        assert entered.wait(3)
        assert controller.active
        assert controller.drain() == []  # Reading the view does not restart HTTP.
        assert controller.active
    finally:
        controller.close()
    assert cancelled.is_set()
    assert not controller.active
    assert not controller.thread.is_alive()


def test_controller_reports_http_failure_and_can_start_a_fresh_reader(monkeypatch):
    app = FastAPI()
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda: real_client(transport=httpx.ASGITransport(app=app)),
    )
    controller = StreamController()
    client = BaseAgentClient("http://fixture", agent="juena")
    for _ in range(2):
        controller.start("thread", client.astream("original"))
        controller.thread.join(timeout=3)
        assert not controller.active
        chunks = controller.drain()
        assert len(chunks) == 1 and chunks[0]["type"] == "error"
        assert "404" in controller.error or "Not Found" in controller.error
    controller.close()
