"""The synchronous HTTP client for the routes core serves.

Split out of ``juena/clients/client.py`` (00-BOUNDARY.md, decision 10). The
application's ``AgentClient`` calls ``/auth/me`` and ``/research``, neither of
which is a core route, so core gets only :class:`BaseAgentClient`: transport,
SSE parsing, and one method per route :func:`juena_core.server.service.create_app`
actually serves. Each application subclasses it and adds its own.

**Synchronous on purpose.** Streamlit consumes it from a script run, and
``stream``/``resume_stream`` are ordinary generators. An async client would
have to be driven by an event loop the page does not have.

Two shapes here are deliberately open, because core does not know every event
or every interrupt an application will define:

``_parse_sse_data``
    Core understands its own wire vocabulary and **passes anything else
    through unchanged**, so an application-specific event reaches that
    application's UI intact. The source flattened unknown frames to
    ``{"type": "unknown", "content": str(payload)}``, which destroyed them.

``resume_stream``
    Takes the interrupt ``kind`` as an argument rather than naming one.
    ``clarification`` is core's because ``ask_user`` is core's; every other
    kind is registered by an application through
    :func:`juena_core.server.interrupts.register_interrupt_kind`, and only
    that application knows what to call it.
"""

from __future__ import annotations

import json
import weakref
from collections.abc import Generator, Mapping, Sequence
from typing import Any, NoReturn

import httpx
from httpx_sse import EventSource, SSEError

from juena_core.schema.llm_models import Provider
from juena_core.schema.server import ChatMessage, StreamInput

__all__ = ["AgentClientError", "AgentAuthenticationError", "BaseAgentClient"]


class AgentClientError(Exception):
    """Any failure talking to the agent service."""


class AgentAuthenticationError(AgentClientError):
    """The API rejected the client's user session.

    Raised for every ``401``, not just the one route that checked for it in
    the source. A caller needs to tell "your session expired, sign in again"
    from "the server is broken", and that distinction is the same on every
    route.
    """


def _provider_value(provider: str | Provider | None) -> str | None:
    if provider is None:
        return None
    if isinstance(provider, Provider):
        return provider.value
    return provider


def _http_error_message(exc: httpx.HTTPError) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if detail:
                return f"Error: {detail}"
    return f"Error: {exc}"


def _raise_http_error(exc: httpx.HTTPError) -> NoReturn:
    """Translate a transport failure into the right client exception."""

    response = getattr(exc, "response", None)
    if response is not None and response.status_code == 401:
        raise AgentAuthenticationError(_http_error_message(exc)) from exc
    raise AgentClientError(_http_error_message(exc)) from exc


class BaseAgentClient:
    """Client for the routes core serves.

    Owns ``health``, ``list_chats``, ``create_chat``, ``get_chat``,
    ``update_chat``, ``stream``, ``resume_stream``, ``get_artifact``,
    ``get_pending_interrupt`` and ``delete_thread``. It does **not** own
    ``get_current_user`` (``/auth/me``) or ``list_research`` (``/research``):
    those routes belong to an application, and a client method with no route
    behind it is a promise core cannot keep.
    """

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        session_token: str | None = None,
    ) -> None:
        """
        Args:
            base_url: Base URL of the agent service.
            agent: The agent id this client streams to.
            timeout: Request timeout in seconds, or ``None`` for no limit.
            session_token: The session cookie value, when the app sets one.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.agent: str | None = agent
        self.session_token = session_token
        self._client = httpx.Client()
        self._client_finalizer = weakref.finalize(self, self._client.close)

    def close(self) -> None:
        """Close the reusable HTTP connection pool."""
        if self._client_finalizer.alive:
            self._client_finalizer()

    def __enter__(self) -> BaseAgentClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def health(self) -> bool:
        """Return whether the API health endpoint is reachable."""
        try:
            response = self._client.get(f"{self.base_url}/health", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    @property
    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.session_token:
            headers["Cookie"] = f"juena_session={self.session_token}"
        return headers

    def _require_agent(self) -> str:
        if not self.agent:
            raise AgentClientError(
                "No agent selected. Construct the client with agent=<agent id>, "
                "or set .agent before streaming."
            )
        return self.agent

    def _stream_timeout(self) -> httpx.Timeout | None:
        """Bound connection setup and writes, but never the read.

        A streamed response legitimately stays open while the agent works, so
        a read timeout would cut off a long turn mid-answer.
        """
        if self.timeout is None:
            return None
        return httpx.Timeout(
            connect=self.timeout,
            read=None,
            write=self.timeout,
            pool=self.timeout,
        )

    def _build_attachment_payloads(
        self,
        attachments: Sequence[Any] | None,
    ) -> list[tuple[str, tuple[str, bytes, str]]]:
        payloads: list[tuple[str, tuple[str, bytes, str]]] = []
        for index, attachment in enumerate(attachments or [], start=1):
            filename = str(getattr(attachment, "name", "") or f"attachment_{index}.txt")
            content_type = str(
                getattr(attachment, "type", None)
                or getattr(attachment, "content_type", None)
                or "text/plain"
            )
            if hasattr(attachment, "getvalue"):
                content = attachment.getvalue()
            elif hasattr(attachment, "read"):
                content = attachment.read()
                seek = getattr(attachment, "seek", None)
                if callable(seek):
                    seek(0)
            else:
                raise AgentClientError(f"Unsupported attachment object for '{filename}'")

            if not isinstance(content, bytes):
                raise AgentClientError(f"Attachment '{filename}' did not provide bytes")
            payloads.append(("attachments", (filename, content, content_type)))
        return payloads

    def _build_file_request_data(
        self,
        message: str,
        model: str | None,
        provider: str | Provider | None,
        thread_id: str | None,
    ) -> dict[str, str]:
        data = {"message": message}
        provider_value = _provider_value(provider)
        if thread_id:
            data["thread_id"] = thread_id
        if model:
            data["model"] = model
        if provider_value:
            data["provider"] = provider_value
        return data

    # ------------------------------------------------------------------
    # Chats
    # ------------------------------------------------------------------

    def list_chats(self, limit: int = 50, *, agent_id: str | None = None) -> list[dict[str, Any]]:
        """List conversations owned by the authenticated user.

        ``agent_id`` narrows the list to one agent. Omitting it lists them
        all, which shows more rather than granting more.
        """
        params: dict[str, Any] = {"limit": limit}
        if agent_id is not None:
            params["agent_id"] = agent_id
        try:
            response = self._client.get(
                f"{self.base_url}/chats",
                params=params,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _raise_http_error(exc)
        payload = response.json()
        if not isinstance(payload, list):
            raise AgentClientError("Error: Invalid chat-list response from server")
        return payload

    def create_chat(
        self,
        thread_id: str,
        *,
        agent_id: str,
        title: str = "New Chat",
    ) -> dict[str, Any]:
        """Create an authenticated, user-owned conversation.

        ``agent_id`` is required and keyword-only. A thread belongs to one
        registered graph for its whole life, and the server has no default to
        fall back on: guessing here would create the row under the wrong
        graph, and every later request for that thread would answer 404.
        """
        try:
            response = self._client.post(
                f"{self.base_url}/chats",
                json={"thread_id": thread_id, "agent_id": agent_id, "title": title},
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _raise_http_error(exc)
        return response.json()

    def get_chat(self, thread_id: str, include_messages: bool = True) -> dict[str, Any] | None:
        """Load an owned conversation, optionally without its message history."""
        try:
            response = self._client.get(
                f"{self.base_url}/chats/{thread_id}",
                params={"include_messages": include_messages},
                headers=self._headers,
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _raise_http_error(exc)
        return response.json()

    def update_chat(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Update metadata on an owned conversation."""
        payload = {
            key: value
            for key, value in {"title": title, "summary": summary}.items()
            if value is not None
        }
        try:
            response = self._client.patch(
                f"{self.base_url}/chats/{thread_id}",
                json=payload,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _raise_http_error(exc)
        return response.json()

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _parse_sse_data(self, data: str) -> ChatMessage | dict[str, Any] | None:
        """Parse one SSE ``data`` payload into a message or an event dict.

        Core's own wire vocabulary is ``message``, ``error``, ``thread``,
        ``token``, ``status``, ``clarification_required``,
        ``approval_required`` and ``artifact``. **Anything else is returned
        as it arrived**, with its ``type`` intact.

        That passthrough is what lets an application define its own event —
        juena-chatbot's ``sandbox_status``, say — without core learning about
        it, and it mirrors the server side, where
        :class:`~juena_core.server.streaming.processor.StreamPolicy` decides
        which custom events are forwarded at all.

        Returns ``None`` only for payloads carrying nothing to render;
        callers must skip those rather than treat them as end-of-stream.
        """
        try:
            parsed = json.loads(data)
        except ValueError as exc:
            raise AgentClientError(f"Error JSON parsing message from server: {exc}") from exc

        if not isinstance(parsed, dict):
            raise AgentClientError("Error: Server sent a stream frame that is not an object")

        parsed_type = parsed.get("type", "")

        if parsed_type == "message":
            try:
                return ChatMessage.model_validate(parsed["content"])
            except Exception as exc:
                raise AgentClientError(f"Server returned invalid message: {exc}") from exc

        if parsed_type == "error":
            return ChatMessage(
                type="ai", content="Error: " + parsed.get("content", "Unknown error")
            )

        if parsed_type == "thread":
            return {"type": "thread", "thread_id": parsed.get("thread_id", "")}

        if parsed_type == "token":
            return {"type": "token", "content": parsed.get("content", "")}

        if parsed_type == "status":
            return {
                "type": "status",
                "phase": parsed.get("phase", ""),
                "label": parsed.get("label", ""),
                "tool": parsed.get("tool"),
                "summary": parsed.get("summary"),
            }

        if parsed_type == "clarification_required":
            return {
                "type": "clarification_required",
                "interrupt_id": parsed.get("interrupt_id", ""),
                "asked_by": parsed.get("asked_by", ""),
                "question": parsed.get("question", ""),
                "options": parsed.get("options") or [],
            }

        if parsed_type == "approval_required":
            return {
                "type": "approval_required",
                "interrupt_id": parsed.get("interrupt_id", ""),
                "action_name": parsed.get("action_name", ""),
                "command": parsed.get("command", ""),
                "description": parsed.get("description", ""),
                "limits": parsed.get("limits") or {},
                "allowed_decisions": parsed.get("allowed_decisions") or [],
            }

        if parsed_type == "artifact":
            return {
                "type": "artifact",
                "artifact_id": parsed.get("artifact_id", ""),
                "filename": parsed.get("filename", "plot.png"),
                "mime_type": parsed.get("mime_type", "image/png"),
                "kind": parsed.get("kind", "image"),
                "size": parsed.get("size"),
                "width": parsed.get("width"),
                "height": parsed.get("height"),
                "caption": parsed.get("caption", "Plot"),
                "created_at": parsed.get("created_at"),
                "content_base64": parsed.get("content_base64", ""),
            }

        # An application's own event. Handed over whole: core cannot render it,
        # and summarising it would leave the application's UI nothing to work
        # from.
        return parsed

    def _build_stream_request(
        self,
        message: str,
        model: str | None,
        provider: str | None,
        thread_id: str | None,
        attachments: Sequence[Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Build the URL and httpx kwargs for a new turn."""
        agent = self._require_agent()
        stream_kwargs: dict[str, Any] = {
            "headers": self._headers,
            "timeout": self._stream_timeout(),
        }

        if attachments:
            stream_kwargs["data"] = self._build_file_request_data(
                message, model, provider, thread_id
            )
            stream_kwargs["files"] = self._build_attachment_payloads(attachments)
            return f"{self.base_url}/{agent}/stream_with_files", stream_kwargs

        request = StreamInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if provider:
            request.provider = Provider(provider) if isinstance(provider, str) else provider
        stream_kwargs["json"] = request.model_dump(exclude_none=True)
        return f"{self.base_url}/{agent}/stream", stream_kwargs

    def _iter_stream(
        self,
        url: str,
        **stream_kwargs: Any,
    ) -> Generator[ChatMessage | dict[str, Any], None, None]:
        """Yield parsed frames from one POSTed SSE response."""
        try:
            with self._client.stream("POST", url, **stream_kwargs) as response:
                response.raise_for_status()
                for sse in EventSource(response).iter_sse():
                    parsed = self._parse_sse_data(sse.data)
                    if parsed is not None:
                        yield parsed
        except httpx.HTTPError as exc:
            _raise_http_error(exc)
        except SSEError as exc:
            raise AgentClientError(f"Error: {exc}") from exc

    def stream(
        self,
        message: str,
        model: str | None = None,
        provider: str | None = None,
        thread_id: str | None = None,
        attachments: Sequence[Any] | None = None,
    ) -> Generator[ChatMessage | dict[str, Any], None, None]:
        """Stream the agent's response synchronously.

        Yields complete messages as :class:`ChatMessage`, and ``token`` /
        ``status`` / ``thread`` dicts as the run progresses.

        Args:
            message: The message to send to the agent.
            model: LLM model to use for this turn.
            provider: LLM provider to use for this turn.
            thread_id: Thread to continue, or ``None`` to start one.
            attachments: Files to stage for this turn.
        """
        url, stream_kwargs = self._build_stream_request(
            message, model, provider, thread_id, attachments
        )
        yield from self._iter_stream(url, **stream_kwargs)

    def resume_stream(
        self,
        *,
        thread_id: str,
        interrupt_id: str,
        kind: str,
        model: str | None = None,
        provider: str | Provider | None = None,
        **fields: Any,
    ) -> Generator[ChatMessage | dict[str, Any], None, None]:
        """Answer a pending interrupt and stream the rest of the turn.

        ``kind`` selects the arm of the resume union the application composed
        for ``create_app(resume_input=...)``, and ``fields`` carries whatever
        that arm requires — ``answer="…"`` for core's ``clarification``, or
        ``decision="approve"`` for an application-registered approval.

        Core names no kind but its own here. ``ask_user`` is core's, so
        ``clarification`` is; an approval belongs to whichever application
        registered it, under whatever name that application chose.
        """
        agent = self._require_agent()
        payload: dict[str, Any] = {
            "thread_id": thread_id,
            "interrupt_id": interrupt_id,
            "kind": kind,
            **{key: value for key, value in fields.items() if value is not None},
        }
        if model:
            payload["model"] = model
        if provider_value := _provider_value(provider):
            payload["provider"] = provider_value

        yield from self._iter_stream(
            f"{self.base_url}/{agent}/resume",
            json=payload,
            headers=self._headers,
            timeout=self._stream_timeout(),
        )

    # ------------------------------------------------------------------
    # Threads and artifacts
    # ------------------------------------------------------------------

    def get_artifact(self, artifact_id: str) -> bytes:
        """Download one generated file owned by the current user."""
        try:
            response = self._client.get(
                f"{self.base_url}/artifacts/{artifact_id}",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _raise_http_error(exc)
        if not response.content:
            raise AgentClientError("Error: Server returned an empty generated file")
        return response.content

    def get_pending_interrupt(
        self,
        thread_id: str,
        *,
        model: str | None = None,
        provider: str | Provider | None = None,
    ) -> dict[str, Any] | None:
        """Return a thread's pending question or approval, if any.

        The payload keeps its own ``type`` (``clarification_required`` or
        ``approval_required``), so callers dispatch on that rather than on
        which slot it arrived under.
        """
        params = {
            key: value
            for key, value in {
                "agent_id": self._require_agent(),
                "model": model,
                "provider": _provider_value(provider),
            }.items()
            if value
        }
        try:
            response = self._client.get(
                f"{self.base_url}/threads/{thread_id}/pending-interrupt",
                params=params,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            _raise_http_error(exc)
        except ValueError as exc:
            raise AgentClientError(
                "Error: Server returned an invalid interrupt response"
            ) from exc
        if not isinstance(payload, dict):
            return None
        pending = payload.get("approval") or payload.get("clarification")
        return pending if isinstance(pending, dict) else None

    def delete_thread(self, thread_id: str) -> dict[str, Any]:
        """Delete all persisted server-side state for a conversation thread."""
        try:
            response = self._client.delete(
                f"{self.base_url}/threads/{thread_id}",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _raise_http_error(exc)
        return response.json()

    # ------------------------------------------------------------------
    # Frame helpers
    # ------------------------------------------------------------------

    def is_token_message(self, message: ChatMessage | str | Mapping[str, Any]) -> bool:
        """Whether a streamed frame is an incremental text chunk."""
        if isinstance(message, str):
            return True  # Legacy string token
        if isinstance(message, Mapping):
            return message.get("type", "") == "token"
        return False

    def get_token_content(self, message: ChatMessage | str | Mapping[str, Any]) -> str | None:
        """Return the text carried by a token frame, or ``None``."""
        if isinstance(message, str):
            return message  # Legacy string token
        if isinstance(message, Mapping) and self.is_token_message(message):
            return message.get("content")
        return None
