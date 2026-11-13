"""The HTTP client for the routes core serves.

:class:`BaseAgentClient` holds the transport, SSE parsing (SSE, server-sent
events, is how the server streams a reply) and one method per route
:func:`juena_core.server.service.create_app` serves. Each application
subclasses it and adds methods for its own routes.

Synchronous by default: Streamlit calls it from a script run, which has no
event loop, so ``stream`` and ``resume_stream`` are ordinary generators.
``astream`` and ``aresume_stream`` are async versions for callers that have
an event loop; each opens its own connection.

Two parts stay open, because core does not know every event or every
interrupt an application will define:

``_parse_sse_data``
    Understands core's own event types and passes anything else through
    unchanged, so an application's own event reaches its UI intact.

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
from collections.abc import AsyncGenerator, Generator, Mapping, Sequence
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

    Raised for every ``401``, so a caller can tell "your session expired, sign
    in again" from "the server is broken" on any route.
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
            if isinstance(detail, dict) and isinstance(detail.get("message"), str):
                detail = detail["message"]
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

    Methods for routes an application adds belong in that application's
    subclass: a client method with no route behind it is a promise core
    cannot keep.
    """

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        session_token: str | None = None,
    ) -> None:
        """Initialize the client.

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
        """Return whether the API health endpoint is reachable.

        Returns:
            Whether ``/health`` answered 200 within two seconds.
        """
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

    def _filesystem_request(self, method: str, thread_id: str, suffix: str = "", **kwargs):
        from urllib.parse import quote
        try:
            response = self._client.request(
                method, f"{self.base_url}/threads/{quote(thread_id, safe='')}/filesystem{suffix}",
                headers=self._headers, timeout=self.timeout, **kwargs,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _raise_http_error(exc)
        return response

    def list_files(self, thread_id: str, path: str = "/", *, hidden: bool = False, offset: int = 0):
        """List one page of a directory in the conversation's files.

        Args:
            thread_id: The conversation.
            path: Directory to list.
            hidden: Whether to include names that start with a dot.
            offset: Index of the first entry to return.

        Returns:
            The server's page: ``path``, ``entries`` and ``next_offset``.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: Any other request failure.
        """
        return self._filesystem_request("GET", thread_id, params={"path": path, "hidden": hidden, "offset": offset}).json()

    def read_file(self, thread_id: str, path: str):
        """Load a preview of one conversation file.

        Args:
            thread_id: The conversation.
            path: The file.

        Returns:
            The server's preview, including the text and the revision to pass
            to :meth:`save_file`.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: Any other request failure.
        """
        return self._filesystem_request("GET", thread_id, "/file", params={"path": path}).json()

    def download_file(self, thread_id: str, path: str) -> bytes:
        """Download one conversation file.

        Args:
            thread_id: The conversation.
            path: The file.

        Returns:
            The file's bytes.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: Any other request failure.
        """
        return self._filesystem_request("GET", thread_id, "/download", params={"path": path}).content

    def save_file(self, thread_id: str, path: str, text: str, revision: str):
        """Save edited text to a workspace file.

        Args:
            thread_id: The conversation.
            path: The file.
            text: The new content.
            revision: The revision the edit started from; the server refuses
                the save when the file has changed since.

        Returns:
            The path and the file's new revision.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: Any other request failure, including a changed
                file.
        """
        return self._filesystem_request("PUT", thread_id, "/file", json={
            "path": path, "text": text, "revision": revision,
        }).json()

    # ------------------------------------------------------------------
    # Chats
    # ------------------------------------------------------------------

    def list_chats(self, limit: int = 50, *, agent_id: str | None = None) -> list[dict[str, Any]]:
        """List conversations owned by the authenticated user.

        ``agent_id`` narrows the list to one agent. Omitting it lists them
        all, which shows more rather than granting more.

        Args:
            limit: Most conversations to return.
            agent_id: Agent to list conversations for; ``None`` means all.

        Returns:
            The conversations as dicts.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: Any other request failure or an invalid response.
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

        Args:
            thread_id: Id for the new conversation.
            agent_id: Agent the conversation belongs to.
            title: The conversation's title.

        Returns:
            The created conversation.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: Any other request failure.
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
        """Load an owned conversation, optionally without its message history.

        Args:
            thread_id: The conversation.
            include_messages: Whether to include the message history.

        Returns:
            The conversation, or ``None`` when the server answers 404 or 410.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: Any other request failure.
        """
        try:
            response = self._client.get(
                f"{self.base_url}/chats/{thread_id}",
                params={"include_messages": include_messages},
                headers=self._headers,
                timeout=self.timeout,
            )
            if response.status_code in (404, 410):
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
        """Update metadata on an owned conversation.

        Args:
            thread_id: The conversation.
            title: New title; ``None`` leaves it unchanged.
            summary: New summary; ``None`` leaves it unchanged.

        Returns:
            The updated conversation.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: Any other request failure.
        """
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

        Core's own event types are ``message``, ``error``, ``thread``,
        ``token``, ``status``, ``clarification_required``,
        ``approval_required`` and ``artifact``. Anything else is returned as
        it arrived, with its ``type`` intact, so an application can define its
        own events without core knowing them. On the server side,
        :class:`~juena_core.server.streaming.processor.StreamPolicy` decides
        which custom events are forwarded at all.

        Core's version never returns ``None``; a subclass may, for a payload
        with nothing to render. Callers skip ``None`` rather than treat it as
        the end of the stream.
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
            raise AgentClientError(parsed.get("content") or "Unknown server error")

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
                "tool_call_id": parsed.get("tool_call_id"),
                "failed": bool(parsed.get("failed")),
                "from_subagent": bool(parsed.get("from_subagent")),
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
            if "actions" in parsed:
                return parsed
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
                # Lets the UI keep only the newest version of a rewritten file.
                "workspace_path": parsed.get("workspace_path"),
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

        Yields:
            Each parsed frame of the response.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: No agent is selected, the request failed, or
                the server sent an error or an invalid frame.
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

        Args:
            thread_id: The conversation.
            interrupt_id: The pending interrupt being answered.
            kind: The interrupt kind, such as ``"clarification"``.
            model: LLM model to use for the rest of the turn.
            provider: LLM provider to use for the rest of the turn.
            **fields: The answer fields for this kind; ``None`` values are
                left out.

        Yields:
            Each parsed frame of the response.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: No agent is selected, the request failed, or
                the server sent an error or an invalid frame.
        """
        url, kwargs = self._build_resume_request(
            thread_id=thread_id, interrupt_id=interrupt_id, kind=kind,
            model=model, provider=provider, **fields,
        )
        yield from self._iter_stream(url, **kwargs)

    async def _aiter_stream(
        self, url: str, **kwargs: Any
    ) -> AsyncGenerator[ChatMessage | dict[str, Any], None]:
        """Own the async connection in the caller's event loop."""
        try:
            async with httpx.AsyncClient() as transport:
                async with transport.stream("POST", url, **kwargs) as response:
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()
                    async for event in EventSource(response).aiter_sse():
                        parsed = self._parse_sse_data(event.data)
                        if parsed is not None:
                            yield parsed
        except httpx.HTTPError as exc:
            _raise_http_error(exc)
        except SSEError as exc:
            raise AgentClientError(f"Error: {exc}") from exc

    async def astream(
        self,
        message: str,
        model: str | None = None,
        provider: str | None = None,
        thread_id: str | None = None,
        attachments: Sequence[Any] | None = None,
    ) -> AsyncGenerator[ChatMessage | dict[str, Any], None]:
        """Stream a turn without sharing the client's synchronous connection.

        Args:
            message: The message to send to the agent.
            model: LLM model to use for this turn.
            provider: LLM provider to use for this turn.
            thread_id: Thread to continue, or ``None`` to start one.
            attachments: Files to stage for this turn.

        Yields:
            Each parsed frame of the response.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: No agent is selected, the request failed, or
                the server sent an error or an invalid frame.
        """
        url, kwargs = self._build_stream_request(
            message, model, provider, thread_id, attachments
        )
        async for chunk in self._aiter_stream(url, **kwargs):
            yield chunk

    async def aresume_stream(
        self,
        *,
        thread_id: str,
        interrupt_id: str,
        kind: str,
        model: str | None = None,
        provider: str | Provider | None = None,
        **fields: Any,
    ) -> AsyncGenerator[ChatMessage | dict[str, Any], None]:
        """Resume an interrupt on a connection owned by the calling event loop.

        Args:
            thread_id: The conversation.
            interrupt_id: The pending interrupt being answered.
            kind: The interrupt kind, such as ``"clarification"``.
            model: LLM model to use for the rest of the turn.
            provider: LLM provider to use for the rest of the turn.
            **fields: The answer fields for this kind; ``None`` values are
                left out.

        Yields:
            Each parsed frame of the response.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: No agent is selected, the request failed, or
                the server sent an error or an invalid frame.
        """
        url, kwargs = self._build_resume_request(
            thread_id=thread_id,
            interrupt_id=interrupt_id,
            kind=kind,
            model=model,
            provider=provider,
            **fields,
        )
        async for chunk in self._aiter_stream(url, **kwargs):
            yield chunk

    def _build_resume_request(
        self, *, thread_id, interrupt_id, kind, model=None, provider=None, **fields
    ):
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

        return f"{self.base_url}/{agent}/resume", {
            "json": payload,
            "headers": self._headers,
            "timeout": self._stream_timeout(),
        }

    def _run_control(self, thread_id, run_id, action, **kwargs):
        url = f"{self.base_url}/threads/{thread_id}/runs/{run_id}{action}"
        try:
            response = self._client.post(
                url, headers=self._headers, timeout=self.timeout, **kwargs
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            _raise_http_error(exc)

    def steer_run(
        self, thread_id: str, run_id: str, instruction_id: str, text: str
    ) -> dict:
        """Send a correction to a running response.

        Args:
            thread_id: The conversation.
            run_id: The running response.
            instruction_id: Id for this correction.
            text: The correction.

        Returns:
            The server's answer.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: Any other request failure.
        """
        return self._run_control(
            thread_id,
            run_id,
            "/steer",
            json={"instruction_id": instruction_id, "text": text},
        )

    def stop_run(self, thread_id: str, run_id: str) -> dict:
        """Ask the server to stop a running response.

        Args:
            thread_id: The conversation.
            run_id: The running response.

        Returns:
            The server's answer.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: Any other request failure.
        """
        return self._run_control(thread_id, run_id, "/stop")

    # ------------------------------------------------------------------
    # Threads and artifacts
    # ------------------------------------------------------------------

    def get_artifact(self, artifact_id: str) -> bytes:
        """Download one generated file owned by the current user.

        Args:
            artifact_id: The file's id.

        Returns:
            The file's bytes.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: Any other request failure or an empty file.
        """
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

        Args:
            thread_id: The conversation.
            model: LLM model, sent along to the server.
            provider: LLM provider, sent along to the server.

        Returns:
            The pending approval or question, or ``None``.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: No agent is selected, the request failed, or the
                response was not JSON.
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
        """Delete all persisted server-side state for a conversation thread.

        Args:
            thread_id: The conversation.

        Returns:
            The server's answer.

        Raises:
            AgentAuthenticationError: The session was rejected.
            AgentClientError: Any other request failure.
        """
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
        """Whether a streamed frame is an incremental text chunk.

        Args:
            message: A frame from :meth:`stream`, or a plain string.

        Returns:
            Whether it is a string or a ``token`` event.
        """
        if isinstance(message, str):
            return True  # Legacy string token
        if isinstance(message, Mapping):
            return message.get("type", "") == "token"
        return False

    def get_token_content(self, message: ChatMessage | str | Mapping[str, Any]) -> str | None:
        """Return the text carried by a token frame, or ``None``.

        Args:
            message: A frame from :meth:`stream`, or a plain string.

        Returns:
            The token text, or ``None`` for any other frame.
        """
        if isinstance(message, str):
            return message  # Legacy string token
        if isinstance(message, Mapping) and self.is_token_message(message):
            return message.get("content")
        return None
