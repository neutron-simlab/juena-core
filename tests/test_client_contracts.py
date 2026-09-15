"""Executable contracts for the synchronous CP5 HTTP client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from juena_core.clients.base import (
    AgentAuthenticationError,
    AgentClientError,
    BaseAgentClient,
)
from juena_core.schema.server import ChatMessage


class _Response:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        content: bytes = b"content",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://api.invalid/test")
            response = httpx.Response(
                self.status_code,
                request=request,
                json=self._payload,
            )
            raise httpx.HTTPStatusError(
                "request failed",
                request=request,
                response=response,
            )

    def json(self) -> Any:
        return self._payload


class _StreamResponse:
    headers = {"content-type": "text/event-stream"}

    def __init__(self, frames: list[str]) -> None:
        self.frames = frames

    def raise_for_status(self) -> None:
        return None

    def iter_text(self):
        for frame in self.frames:
            yield f"data: {frame}\n\n"


class _StreamContext:
    def __init__(self, response: _StreamResponse) -> None:
        self.response = response

    def __enter__(self) -> _StreamResponse:
        return self.response

    def __exit__(self, *_exc_info: object) -> bool:
        return False


@pytest.fixture
def client() -> BaseAgentClient:
    value = BaseAgentClient(
        base_url="http://api.invalid",
        agent="advanced_mode",
        timeout=7.0,
        session_token="opaque-session",
    )
    yield value
    value.close()


def test_client_surface_contains_only_core_routes() -> None:
    expected = {
        "health",
        "list_chats",
        "create_chat",
        "get_chat",
        "update_chat",
        "stream",
        "resume_stream",
        "get_artifact",
        "get_pending_interrupt",
        "delete_thread",
    }
    assert expected <= set(dir(BaseAgentClient))
    assert not hasattr(BaseAgentClient, "get_current_user")
    assert not hasattr(BaseAgentClient, "list_research")


def test_chat_calls_carry_agent_identity_and_session(
    client: BaseAgentClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def get(url: str, **kwargs: Any) -> _Response:
        calls.append(("GET", url, kwargs))
        return _Response([])

    def post(url: str, **kwargs: Any) -> _Response:
        calls.append(("POST", url, kwargs))
        return _Response(kwargs["json"])

    monkeypatch.setattr(client._client, "get", get)
    monkeypatch.setattr(client._client, "post", post)

    client.list_chats(12, agent_id="simulator")
    created = client.create_chat(
        "thread-1",
        agent_id="advanced_mode",
        title="A run",
    )

    assert created == {
        "thread_id": "thread-1",
        "agent_id": "advanced_mode",
        "title": "A run",
    }
    assert calls == [
        (
            "GET",
            "http://api.invalid/chats",
            {
                "params": {"limit": 12, "agent_id": "simulator"},
                "headers": {"Cookie": "juena_session=opaque-session"},
                "timeout": 7.0,
            },
        ),
        (
            "POST",
            "http://api.invalid/chats",
            {
                "json": {
                    "thread_id": "thread-1",
                    "agent_id": "advanced_mode",
                    "title": "A run",
                },
                "headers": {"Cookie": "juena_session=opaque-session"},
                "timeout": 7.0,
            },
        ),
    ]


def test_pending_interrupt_lookup_is_scoped_to_the_clients_agent(
    client: BaseAgentClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def get(url: str, **kwargs: Any) -> _Response:
        captured.update(url=url, **kwargs)
        return _Response(
            {
                "approval": None,
                "clarification": {
                    "type": "clarification_required",
                    "interrupt_id": "interrupt-1",
                },
            }
        )

    monkeypatch.setattr(client._client, "get", get)

    pending = client.get_pending_interrupt(
        "thread-1",
        provider="openai",
        model="gpt-5-mini",
    )

    assert pending == {
        "type": "clarification_required",
        "interrupt_id": "interrupt-1",
    }
    assert captured["url"] == (
        "http://api.invalid/threads/thread-1/pending-interrupt"
    )
    assert captured["params"] == {
        "agent_id": "advanced_mode",
        "provider": "openai",
        "model": "gpt-5-mini",
    }


def test_pending_interrupt_lookup_requires_an_agent() -> None:
    client = BaseAgentClient(base_url="http://api.invalid")
    try:
        with pytest.raises(AgentClientError, match="No agent selected"):
            client.get_pending_interrupt("thread-1")
    finally:
        client.close()


def test_every_unauthorized_response_has_a_distinct_exception(
    client: BaseAgentClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        client._client,
        "get",
        lambda *_args, **_kwargs: _Response(
            {"detail": "Invalid or expired session"},
            status_code=401,
        ),
    )

    with pytest.raises(AgentAuthenticationError, match="Invalid or expired session"):
        client.get_artifact("artifact-1")


def test_stream_uses_multipart_and_leaves_the_read_timeout_open(
    client: BaseAgentClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def stream(method: str, url: str, **kwargs: Any) -> _StreamContext:
        captured.update(method=method, url=url, kwargs=kwargs)
        return _StreamContext(
            _StreamResponse(
                [
                    '{"type":"message","content":{"type":"ai",'
                    '"content":"ok"}}'
                ]
            )
        )

    monkeypatch.setattr(client._client, "stream", stream)
    attachment = SimpleNamespace(
        name="input.dat",
        type="text/plain",
        getvalue=lambda: b"1 2 3\n",
    )

    chunks = list(
        client.stream(
            "simulate",
            provider="openai",
            model="gpt-5-mini",
            thread_id="thread-1",
            attachments=[attachment],
        )
    )

    assert chunks == [ChatMessage(type="ai", content="ok")]
    assert captured["method"] == "POST"
    assert captured["url"] == (
        "http://api.invalid/advanced_mode/stream_with_files"
    )
    assert captured["kwargs"]["data"] == {
        "message": "simulate",
        "thread_id": "thread-1",
        "model": "gpt-5-mini",
        "provider": "openai",
    }
    assert captured["kwargs"]["files"] == [
        ("attachments", ("input.dat", b"1 2 3\n", "text/plain"))
    ]
    timeout = captured["kwargs"]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 7.0
    assert timeout.read is None


def test_resume_stream_keeps_interrupt_kind_application_defined(
    client: BaseAgentClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def stream(method: str, url: str, **kwargs: Any) -> _StreamContext:
        captured.update(method=method, url=url, kwargs=kwargs)
        return _StreamContext(_StreamResponse(['{"type":"token","content":"done"}']))

    monkeypatch.setattr(client._client, "stream", stream)

    chunks = list(
        client.resume_stream(
            thread_id="thread-1",
            interrupt_id="interrupt-1",
            kind="execute_approval",
            decision="edit",
            edited_command="run --safe",
        )
    )

    assert chunks == [{"type": "token", "content": "done"}]
    assert captured["url"] == "http://api.invalid/advanced_mode/resume"
    assert captured["kwargs"]["json"] == {
        "thread_id": "thread-1",
        "interrupt_id": "interrupt-1",
        "kind": "execute_approval",
        "decision": "edit",
        "edited_command": "run --safe",
    }


def test_sse_parser_preserves_unknown_application_events(
    client: BaseAgentClient,
) -> None:
    frame = {
        "type": "simulation_progress",
        "module": "solver",
        "percent": 80,
    }
    assert client._parse_sse_data(
        '{"type":"simulation_progress","module":"solver","percent":80}'
    ) == frame


def test_sse_parser_rejects_non_object_frames(client: BaseAgentClient) -> None:
    with pytest.raises(AgentClientError, match="not an object"):
        client._parse_sse_data('["token"]')
