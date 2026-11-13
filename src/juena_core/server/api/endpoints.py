"""Agent invocation, streaming, resume, thread deletion and artifact download.

This module must not use ``from __future__ import annotations``. The
``/resume`` route annotates its body with the resume union the application
passes to ``build_api_router``, and FastAPI reads that annotation to parse the
request. A deferred annotation is a string looked up in the module's globals,
where that function argument does not exist.

The application passes in what is its own: its resume union, its workspace,
its stream policy, and the note closing a staged-input manifest.
"""

import asyncio
import json
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from juena_core.filesystem.service import ConversationFilesystem
from juena_core.filesystem.storage import WorkspaceBusy
from juena_core.artifacts import get_artifact_store
from juena_core.config import settings
from juena_core.log import get_logger
from juena_core.schema.interrupts import ClarificationResumeInput
from juena_core.schema.llm_models import Provider
from juena_core.schema.server import StreamInput
from juena_core.schema.upload_limits import UploadPolicy
from juena_core.server.agent.input_handler import AgentInputHandler
from juena_core.server.agent.registry import get_agent, get_default_agent
from juena_core.server.chat.inputs import prepare_code_chat_turn_inputs
from juena_core.server.chat.repository import (
    ChatNotFoundError,
    ChatDeletedError,
    ensure_owned_chat,
    get_owned_chat,
)
from juena_core.server.database.checkpointer import get_checkpointer
from juena_core.server.database.connection import get_db_session
from juena_core.server.database.models import utc_now
from juena_core.server.errors import AgentNotFoundError, StateError, StreamingError
from juena_core.server.identity import Principal, PrincipalDependency
from juena_core.server.interrupts import (
    ResumeError,
    build_resume_command,
    first_pending_interrupt,
    interrupt_event,
)
from juena_core.server.streaming import events
from juena_core.server.streaming.own_task import AgentStreamResponse
from juena_core.server.streaming.processor import StreamEventProcessor, StreamPolicy
from juena_core.server.run_control import RunRegistry

logger = get_logger(__name__)

__all__ = [
    "ThreadActivity",
    "ThreadWorkspace",
    "DEFAULT_CLOSING_NOTE",
    "build_api_router",
]

#: Closing paragraph of a staged-input manifest when an application supplies
#: none. It says only what is true for every application: what an agent can do
#: with these paths differs, so an application that wants more passes its own.
DEFAULT_CLOSING_NOTE = "The user's pasted or uploaded materials are staged under `/inputs`."


@dataclass(frozen=True, slots=True)
class ThreadWorkspace:
    """Where a thread's staged files are materialised outside graph state.

    Graph state holds a thread's ``/inputs`` files, which is enough for a model
    that reads them through a file tool, but not for a separate process that
    opens them, such as a sandbox or a program reading from disk. Core computes
    the merged file set and calls these; it never learns where the files land.

    Both are optional. An application with no such process passes neither.

    Attributes:
        stage: ``(user_id, thread_id, files) -> None``, awaited before each run.
        delete: ``(user_id, thread_id) -> None``, awaited when a thread is
            deleted.
    """

    stage: Callable[..., Awaitable[None]] | None = None
    delete: Callable[..., Awaitable[None]] | None = None


class ThreadActivity:
    """Coordinate thread writers with deletion inside one API process."""

    def __init__(self) -> None:
        """Initialize with no active runs and no deletions."""

        self._lock = threading.Lock()
        self._active: dict[str, int] = {}
        self._deleting: set[str] = set()

    def reserve_run(self, thread_id: str, *, exclusive: bool = False) -> bool:
        """Reserve a thread for one run.

        Args:
            thread_id: The conversation thread.
            exclusive: Refuse if the thread already has an active run.

        Returns:
            True if reserved; False if the thread is being deleted, or already
            running when ``exclusive`` is set.
        """

        with self._lock:
            if thread_id in self._deleting or (exclusive and self._active.get(thread_id, 0)):
                return False
            self._active[thread_id] = self._active.get(thread_id, 0) + 1
            return True

    def release_run(self, thread_id: str) -> None:
        """Release one run's reservation on a thread.

        Args:
            thread_id: The conversation thread.
        """

        with self._lock:
            count = self._active.get(thread_id, 0)
            if count <= 1:
                self._active.pop(thread_id, None)
            else:
                self._active[thread_id] = count - 1

    def reserve_delete(self, thread_id: str) -> bool:
        """Reserve a thread for deletion.

        Args:
            thread_id: The conversation thread.

        Returns:
            True if reserved; False if the thread is already being deleted or
            has an active run.
        """

        with self._lock:
            if thread_id in self._deleting or self._active.get(thread_id, 0):
                return False
            self._deleting.add(thread_id)
            return True

    def release_delete(self, thread_id: str) -> None:
        """End a thread's deletion reservation.

        Args:
            thread_id: The conversation thread.
        """

        with self._lock:
            self._deleting.discard(thread_id)


def _sse(payload: dict[str, Any]) -> str:
    """Serialise one SSE payload; ``EventSourceResponse`` adds the framing."""

    return json.dumps(payload)


def _sse_response_example() -> dict[int | str, Any]:
    """SSE response example for the OpenAPI schema."""

    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"type": "status", "phase": "thinking", "label": "Thinking…"}\n\n'
                        'data: {"type": "token", "content": "Hello"}\n\n'
                        'data: {"type": "token", "content": " World"}\n\n'
                    ),
                    "schema": {"type": "string"},
                }
            },
        }
    }


def _form_field(value: str | None) -> str | None:
    """Normalize multipart form fields so empty strings become ``None``."""

    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_agent_id(agent_id: str | None) -> str:
    """The agent this request addresses, or the registered default.

    A missing default raises ``RuntimeError`` rather than answering 404: the
    fault is the application's missing agent import, not the request.
    """

    return agent_id or get_default_agent()


@dataclass(frozen=True)
class PreparedAgentInvocation:
    """Prepared agent kwargs plus the effective injected user message.

    Attributes:
        kwargs: Keyword arguments for ``agent.astream``: input, config and
            context.
        run_id: The id of this run.
        effective_user_message: The text the user typed; the stream processor
            uses it to drop the echoed human message.
    """

    kwargs: dict[str, Any]
    run_id: Any
    effective_user_message: str


class SteeringInput(BaseModel):
    instruction_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    text: str = Field(min_length=1, max_length=32000)


def build_api_router(
    principal: PrincipalDependency,
    *,
    invocation_guard: Callable[[Principal, str | None], Awaitable[None]] | None = None,
    invocation_error: Callable[[Exception], str | None] | None = None,
    resume_input: type[BaseModel] = ClarificationResumeInput,
    workspace: ThreadWorkspace | None = None,
    filesystem: ConversationFilesystem | None = None,
    thread_activity: ThreadActivity | None = None,
    stream_policy: StreamPolicy | None = None,
    closing_note: str = DEFAULT_CLOSING_NOTE,
    allowed_suffixes: Collection[str] | None = None,
    upload_policy: UploadPolicy | None = None,
    run_controls: bool = False,
) -> APIRouter:
    """Build the agent-invocation router against one application's seams.

    Args:
        principal: The identity dependency every route authenticates with.
        invocation_guard: Optional async application readiness check, receiving
            the principal and requested provider. Read-only routes bypass it.
        invocation_error: Optional mapper for safe application-specific error messages;
            return None to retain the standard error handling.
        resume_input: The application's discriminated resume union. The
            default accepts only a clarification reply, for an application
            with no other interrupt kind.
        workspace: Where staged files are materialised, if anywhere.
        filesystem: Shared persistent conversation files, if the application
            has them. Required for binary uploads.
        thread_activity: Guard shared with application routes that can write a
            thread workspace while deletion is in progress.
        stream_policy: Custom event types and silent tools this application's
            execution machinery contributes.
        closing_note: Last paragraph of a staged-input manifest.
        allowed_suffixes: Legacy text upload extensions this application accepts.
        upload_policy: Optional text/binary upload policy; mutually exclusive with allowed_suffixes.
        run_controls: Add the run status, steer and stop routes, and allow only
            one run per thread at a time.

    Returns:
        The router, ready for ``app.include_router``.

    Raises:
        ValueError: Both ``upload_policy`` and ``allowed_suffixes`` are given,
            or binary uploads are allowed without a ``filesystem``.
    """

    if upload_policy is not None and allowed_suffixes is not None:
        raise ValueError("Specify upload_policy or allowed_suffixes, not both")
    if upload_policy and upload_policy.binary_suffixes and filesystem is None:
        raise ValueError("Binary uploads require a conversation filesystem")
    router = APIRouter()
    workspace = workspace or ThreadWorkspace()
    thread_activity = thread_activity or ThreadActivity()
    policy = stream_policy or StreamPolicy()
    runs = RunRegistry()
    controls_router = APIRouter()

    async def _owned_run(thread_id, run_id, user, session):
        """Return the caller's live run on their own chat, or answer 404."""
        try:
            await get_owned_chat(session, user.id, thread_id, agent_id=None)
        except ChatNotFoundError as exc:
            raise HTTPException(404, "Chat not found") from exc
        control = runs.get(run_id, thread_id, str(user.id))
        if control is None:
            raise HTTPException(404, "Run not found")
        return control

    @controls_router.get("/threads/{thread_id}/runs/{run_id}")
    async def run_status(thread_id: str, run_id: str, user: Principal = Depends(principal), session: AsyncSession = Depends(get_db_session)):
        return (await _owned_run(thread_id, run_id, user, session)).snapshot()

    @controls_router.post("/threads/{thread_id}/runs/{run_id}/steer", status_code=202)
    async def steer_run(thread_id: str, run_id: str, payload: SteeringInput, user: Principal = Depends(principal), session: AsyncSession = Depends(get_db_session)):
        control = await _owned_run(thread_id, run_id, user, session)
        if not payload.text.strip():
            raise HTTPException(422, "Instruction text cannot be blank")
        try:
            return control.steer(payload.instruction_id, payload.text)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @controls_router.post("/threads/{thread_id}/runs/{run_id}/stop", status_code=202)
    async def stop_run(thread_id: str, run_id: str, user: Principal = Depends(principal), session: AsyncSession = Depends(get_db_session)):
        control = await _owned_run(thread_id, run_id, user, session)
        control.stop()
        return control.snapshot()

    if run_controls:
        router.include_router(controls_router)

    def _response(source_factory, thread_id, user_id, lease):
        """Wrap a run in a response that owns it and releases the thread at the end."""
        if not run_controls:
            return AgentStreamResponse(source_factory(None), on_close=partial(_release_run, thread_id, lease))
        control = runs.create(thread_id=thread_id, user_id=user_id)
        control.state = "preparing"

        def release():
            if not control.cleanup_failed:
                _release_run(thread_id, lease)

        return AgentStreamResponse(
            source_factory(control), control=control, on_close=release,
            cleanup_seconds=25,
        )

    async def _stream_agent_payloads(
        *,
        agent: CompiledStateGraph,
        kwargs: dict[str, Any],
        run_id: Any,
        authenticated_user_id: str,
        effective_user_message: str,
        ignored_interrupt_ids: set[str] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream one new or resumed graph invocation using LangGraph v2 events."""

        processor = StreamEventProcessor(
            agent,
            kwargs["config"],
            str(run_id),
            effective_user_message,
            include_tool_payloads=settings().STREAM_TOOL_PAYLOADS,
            ignored_interrupt_ids=ignored_interrupt_ids,
            policy=policy,
        )
        thread_id = str(kwargs["config"]["configurable"]["thread_id"])
        artifact_store = get_artifact_store()
        if filesystem is not None:
            files = filesystem.for_thread(authenticated_user_id, thread_id)
            async with files.aguard():
                files.ensure()
                # Protect a checkpointed approval even if this process is killed
                # before reconciliation. Normal completion/cancellation clears it.
                files.store.mark_paused(files.workspace_id, True)
        try:
            yield events.status_event(events.PHASE_THINKING, "Thinking…")

            async for stream_event in agent.astream(
                **kwargs,
                stream_mode=["updates", "messages", "custom"],
                subgraphs=True,
                version="v2",
            ):
                async for payload in processor.process_event(stream_event):
                    yield payload
                for artifact in artifact_store.drain_events(authenticated_user_id, thread_id):
                    yield artifact

        finally:
            if filesystem is not None:
                pending = await first_pending_interrupt(agent, kwargs["config"])
                files.store.mark_paused(files.workspace_id, pending is not None)
        for artifact in artifact_store.drain_events(authenticated_user_id, thread_id):
            yield artifact

    def _release_run(thread_id, lease):
        if lease is not None:
            lease.__exit__(None, None, None)
        thread_activity.release_run(thread_id)

    async def _file_lease(user_id, thread_id):
        """Take a shared lease on the thread's files, if any; a busy workspace answers 409."""
        if filesystem is None:
            return None
        try:
            lease = await filesystem.for_thread(user_id, thread_id).acquire(operation=True, shared=True)
        except WorkspaceBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        return lease

    async def _authorize_thread(
        session: AsyncSession,
        user: Principal,
        thread_id: str | None,
        agent_id: str,
    ) -> str:
        """Create or authorize a thread and return the id the agent must run on.

        Every invocation route calls this, so ownership is checked in one
        place. A plain helper, not a ``Depends``, because ``thread_id`` arrives
        in the request body or form, not the path or query.

        The agent is checked as well as the owner: running a thread under a
        different graph would hand it a checkpoint that does not fit its state.
        Both mismatches answer 404, because saying that a chat exists but is
        not the caller's already tells them something.
        """

        try:
            chat = await ensure_owned_chat(session, user.id, thread_id, agent_id, filesystem=filesystem)
        except ChatDeletedError as exc:
            raise HTTPException(410, {"code": "conversation_deleted", "message": "This conversation was deleted. Start a new chat."}) from exc
        except ChatNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        chat.updated_at = utc_now()
        await session.commit()
        return chat.thread_id

    async def _reserve_authorized_thread(
        session: AsyncSession,
        user: Principal,
        thread_id: str | None,
        agent_id: str,
    ) -> tuple[str, Any]:
        """Hold the activity guard across ownership validation and the run."""

        candidate = thread_id or str(uuid4())
        if not thread_activity.reserve_run(candidate, exclusive=run_controls):
            raise HTTPException(status_code=409, detail="This conversation is already running or being deleted")
        try:
            lease = await _file_lease(str(user.id), candidate)
            try:
                return await _authorize_thread(session, user, candidate, agent_id), lease
            except BaseException:
                if lease is not None:
                    lease.__exit__(None, None, None)
                raise
        except BaseException:
            thread_activity.release_run(candidate)
            raise

    async def _prepare_agent_invocation(
        *,
        agent: CompiledStateGraph,
        message: str,
        thread_id: str | None,
        user_id: str,
        provider: str | None,
        model: str | None,
        attachments: Sequence[UploadFile] | None = None,
        control: Any = None,
    ) -> PreparedAgentInvocation:
        """Prepare LangGraph invocation kwargs for JSON and multipart requests."""

        # The config is built first because staging uploads has to read the
        # thread's existing `/inputs` state, and that read must use the config
        # the run will use.
        run = AgentInputHandler.build_run_context(
            thread_id=thread_id,
            user_id=user_id,
            provider=provider,
            model=model,
            run_id=UUID(control.id) if control is not None else None,
            control=control,
        )
        if control is not None:
            control.agent, control.config = agent, run.config
            control.check()
            control.state = "running"
            control.emit()

        if await first_pending_interrupt(agent, run.config) is not None:
            raise ValueError("Answer the pending question or approval before sending a new message")

        prepared = await prepare_code_chat_turn_inputs(
            agent,
            run.config,
            message,
            attachments=attachments,
            closing_note=closing_note,
            allowed_suffixes=allowed_suffixes,
            upload_policy=upload_policy,
            thread_files=filesystem.for_thread(user_id, run.thread_id) if filesystem else None,
        )
        if workspace.stage is not None:
            await workspace.stage(
                user_id=user_id,
                thread_id=run.thread_id,
                files=(prepared.workspace_files or {}) if prepared is not None else {},
            )

        kwargs = {
            "input": AgentInputHandler.build_input(
                message,
                message_override=(prepared.message_override if prepared is not None else None),
                initial_files=(
                    (prepared.files_update or None) if prepared is not None else None
                ),
                display_attachments=(prepared.attachments if prepared is not None else None),
            ),
            "config": run.config,
            "context": run.context,
        }

        return PreparedAgentInvocation(
            kwargs=kwargs,
            run_id=run.run_id,
            # The raw text, not the manifest: the stream processor drops the
            # echoed human message when it matches this, and that message
            # displays as what the user typed.
            effective_user_message=message,
        )

    async def message_generator(
        user_input: StreamInput,
        authenticated_user_id: str,
        agent_id: str,
        attachments: Sequence[UploadFile] | None = None,
        control: Any = None,
    ) -> AsyncGenerator[str, None]:
        """Generate a stream of SSE payloads from the agent."""

        provider = user_input.provider.value if user_input.provider else None
        model = user_input.model

        try:
            agent: CompiledStateGraph = await get_agent(agent_id, provider=provider, model=model)
        except AgentNotFoundError as e:
            logger.error(f"Agent not found: {e}")
            yield _sse(events.error_event(f"Agent not found: {e.message}"))
            return

        try:
            prepared = await _prepare_agent_invocation(
                agent=agent,
                message=user_input.message,
                thread_id=user_input.thread_id,
                user_id=authenticated_user_id,
                provider=provider,
                model=model,
                attachments=attachments,
                control=control,
            )
            kwargs = prepared.kwargs
            run_id = prepared.run_id
        except StateError as e:
            logger.error(f"Failed to prepare input: {e}")
            yield _sse(events.error_event(f"Failed to prepare input: {e.message}"))
            return
        except ValueError as e:
            logger.error(f"Invalid request input: {e}")
            yield _sse(events.error_event(str(e)))
            return
        except Exception:
            logger.exception("Unexpected error preparing input")
            yield _sse(events.error_event("Unexpected error preparing input"))
            return

        # Emit thread_id first so clients can use it for follow-up messages.
        thread_id_used = kwargs.get("config", {}).get("configurable", {}).get("thread_id")
        if thread_id_used:
            if control is None:
                yield _sse(events.thread_event(thread_id_used))
            # Discard artifacts a cancelled or crashed previous turn left
            # undrained. Here, not inside the graph: a turn paused for approval
            # continues in a second graph run (the resume), which must keep them.
            get_artifact_store().begin_turn(authenticated_user_id, thread_id_used)

        # Messages are not persisted here: the checkpointer commits the full
        # thread history as the graph runs, and `/chats/{thread_id}` reads it back.
        try:
            async for payload in _stream_agent_payloads(
                agent=agent,
                kwargs=kwargs,
                run_id=run_id,
                authenticated_user_id=authenticated_user_id,
                effective_user_message=prepared.effective_user_message,
            ):
                yield _sse(payload)

        except asyncio.CancelledError:
            # Client disconnected or request cancelled; re-raise so the task is
            # properly cancelled.
            logger.debug("Stream cancelled (client disconnect or request cancelled)")
            raise
        except StreamingError as e:
            logger.error(f"Streaming error: {e}")
            yield _sse(events.error_event(f"Streaming error: {e.message}"))
        except Exception as exc:
            safe_message = invocation_error(exc) if invocation_error else None
            if safe_message is None:
                logger.exception("Unexpected error in message generator")
            yield _sse(events.error_event(safe_message or "Internal server error"))

    async def resume_generator(
        payload: Any,
        authenticated_user_id: str,
        agent_id: str,
        control: Any = None,
    ) -> AsyncGenerator[str, None]:
        """Resume one verified LangGraph interrupt and stream the remaining turn."""

        try:
            agent = await get_agent(agent_id, provider=payload.provider, model=payload.model)
            run = AgentInputHandler.build_run_context(
                thread_id=payload.thread_id,
                user_id=authenticated_user_id,
                provider=payload.provider,
                model=payload.model,
                run_id=UUID(control.id) if control is not None else None,
                control=control,
            )
            if control is not None:
                control.agent, control.config = agent, run.config
                control.check()
                control.state = "running"
                control.emit()
            command = await build_resume_command(
                agent=agent,
                config=run.config,
                user_id=authenticated_user_id,
                payload=payload,
            )
            kwargs = {"input": command, "config": run.config, "context": run.context}
            if control is None:
                yield _sse(events.thread_event(payload.thread_id))
            async for item in _stream_agent_payloads(
                agent=agent,
                kwargs=kwargs,
                run_id=run.run_id,
                authenticated_user_id=authenticated_user_id,
                effective_user_message="",
                ignored_interrupt_ids={payload.interrupt_id},
            ):
                yield _sse(item)
        except ResumeError as exc:
            yield _sse(events.error_event(str(exc)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            safe_message = invocation_error(exc) if invocation_error else None
            if safe_message is None:
                logger.exception("Unexpected error resuming an interrupted turn")
            yield _sse(events.error_event(safe_message or "Unable to resume this turn"))

    @router.post(
        "/{agent_id}/stream",
        response_class=EventSourceResponse,
        responses=_sse_response_example(),
    )
    @router.post("/stream", response_class=EventSourceResponse, responses=_sse_response_example())
    async def stream(
        user_input: StreamInput,
        agent_id: str | None = None,
        user: Principal = Depends(principal),
        session: AsyncSession = Depends(get_db_session),
    ) -> EventSourceResponse:
        """Stream an agent's response, including intermediate messages and tokens.

        Use ``thread_id`` to continue a multi-turn conversation; when it is
        absent a new thread is created and reported by the first event.
        """

        if invocation_guard is not None:
            await invocation_guard(user, user_input.provider)
        resolved = _resolve_agent_id(agent_id)
        user_input.thread_id, lease = await _reserve_authorized_thread(
            session, user, user_input.thread_id, resolved
        )
        # The response owns the run: however it ends, the agent is stopped
        # and cleaned up before the thread is released.
        return _response(
            lambda control: message_generator(user_input, str(user.id), resolved, control=control),
            user_input.thread_id, str(user.id), lease,
        )

    @router.post(
        "/{agent_id}/stream_with_files",
        response_class=EventSourceResponse,
        responses=_sse_response_example(),
    )
    @router.post(
        "/stream_with_files",
        response_class=EventSourceResponse,
        responses=_sse_response_example(),
    )
    async def stream_with_files(
        agent_id: str | None = None,
        message: str = Form(""),
        thread_id: str | None = Form(None),
        provider: str | None = Form(None),
        model: str | None = Form(None),
        attachments: list[UploadFile] | None = File(default=None),
        user: Principal = Depends(principal),
        session: AsyncSession = Depends(get_db_session),
    ) -> EventSourceResponse:
        """Stream an agent response from multipart form data and validated attachments."""

        user_input = StreamInput(message=message, thread_id=_form_field(thread_id))
        if (provider_value := _form_field(provider)) is not None:
            try:
                user_input.provider = Provider(provider_value)
            except ValueError as e:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported provider: {provider_value}"
                ) from e
        if (model_value := _form_field(model)) is not None:
            user_input.model = model_value
        if invocation_guard is not None:
            await invocation_guard(user, user_input.provider)
        resolved = _resolve_agent_id(agent_id)
        user_input.thread_id, lease = await _reserve_authorized_thread(
            session, user, user_input.thread_id, resolved
        )
        return _response(
            lambda control: message_generator(
                user_input,
                str(user.id),
                resolved,
                attachments=attachments,
                control=control,
            ),
            user_input.thread_id, str(user.id), lease,
        )

    @router.post(
        "/{agent_id}/resume",
        response_class=EventSourceResponse,
        responses=_sse_response_example(),
    )
    @router.post("/resume", response_class=EventSourceResponse, responses=_sse_response_example())
    async def resume(
        payload: resume_input,
        agent_id: str | None = None,
        user: Principal = Depends(principal),
        session: AsyncSession = Depends(get_db_session),
    ) -> EventSourceResponse:
        """Resume a pending interrupt after an authenticated reply."""

        if invocation_guard is not None:
            await invocation_guard(user, getattr(payload, "provider", None))
        resolved = _resolve_agent_id(agent_id)
        if not thread_activity.reserve_run(payload.thread_id, exclusive=run_controls):
            raise HTTPException(status_code=409, detail="This conversation is already running or being deleted")
        lease = None
        try:
            lease = await _file_lease(str(user.id), payload.thread_id)
            await get_owned_chat(session, user.id, payload.thread_id, agent_id=resolved)
        except ChatNotFoundError as exc:
            _release_run(payload.thread_id, lease)
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        except BaseException:
            _release_run(payload.thread_id, lease)
            raise
        return _response(
            lambda control: resume_generator(payload, str(user.id), resolved, control=control),
            payload.thread_id, str(user.id), lease,
        )

    @router.get("/threads/{thread_id}/pending-approval")
    @router.get("/threads/{thread_id}/pending-interrupt")
    async def pending_interrupt(
        thread_id: str,
        agent_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        user: Principal = Depends(principal),
        session: AsyncSession = Depends(get_db_session),
    ) -> dict[str, Any]:
        """Restore a pending question or approval after a UI reload.

        ``clarification`` carries a pending question; ``approval`` carries an
        application-registered approval, and stays in the payload so an older
        client keeps working.
        """

        resolved = _resolve_agent_id(agent_id)
        try:
            await get_owned_chat(session, user.id, thread_id, agent_id=resolved)
        except ChatNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc

        agent = await get_agent(resolved, provider=provider, model=model)
        run = AgentInputHandler.build_run_context(
            thread_id=thread_id,
            user_id=str(user.id),
            provider=provider,
            model=model,
        )
        empty: dict[str, Any] = {"approval": None, "clarification": None}
        try:
            pending = await first_pending_interrupt(agent, run.config)
        except ResumeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if pending is None:
            return empty
        interrupt, _kind = pending
        payload = interrupt_event(interrupt)
        if payload is None:
            return empty
        # A client that only knows approvals still finds one under "approval".
        slot = "clarification" if payload["type"] == "clarification_required" else "approval"
        return {**empty, slot: payload}

    @router.delete("/threads/{thread_id}")
    async def delete_thread(
        thread_id: str,
        user: Principal = Depends(principal),
        session: AsyncSession = Depends(get_db_session),
    ) -> dict[str, Any]:
        """Delete all persisted state for a conversation thread.

        No agent is required: the path carries none, and a caller may delete a
        conversation of theirs whichever agent owns it.
        """

        try:
            chat = await get_owned_chat(session, user.id, thread_id, agent_id=None)
        except ChatNotFoundError as e:
            raise HTTPException(status_code=404, detail="Chat not found") from e

        if not thread_activity.reserve_delete(thread_id):
            raise HTTPException(status_code=409, detail="Thread has an active operation")

        file_guard = None
        try:
            if filesystem is not None:
                file_guard = filesystem.for_thread(str(user.id), thread_id).guard(operation=True)
                try:
                    file_guard.__enter__()
                    files = filesystem.for_thread(str(user.id), thread_id)
                    await asyncio.to_thread(filesystem.assert_idle, files.workspace_id)
                except WorkspaceBusy as exc:
                    file_guard.__exit__(None, None, None)
                    file_guard = None
                    raise HTTPException(409, str(exc)) from exc
            # The first ownership lookup may have raced with another completed
            # deletion before this request acquired the guard. Recheck while
            # holding it so stale authorization cannot clean up a vanished row.
            try:
                chat = await get_owned_chat(session, user.id, thread_id, agent_id=None)
            except ChatNotFoundError as e:
                raise HTTPException(status_code=404, detail="Chat not found") from e

            checkpointer = get_checkpointer()
            artifact_store = get_artifact_store()
            if workspace.delete is not None:
                await workspace.delete(user_id=str(user.id), thread_id=thread_id)
            if filesystem is not None:
                files = filesystem.for_thread(str(user.id), thread_id)
                async with files.aguard():
                    await asyncio.to_thread(files.store.delete, files.workspace_id, files.tenant_id)
            await checkpointer.adelete_thread(thread_id)
            artifact_store.delete_thread(str(user.id), thread_id)
            await session.delete(chat)
            await session.commit()
            return {
                "status": "success",
                "thread_id": thread_id,
                "message": f"Thread {thread_id} deleted successfully",
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to delete thread %s: %s", thread_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to delete thread state") from e
        finally:
            if file_guard is not None:
                file_guard.__exit__(None, None, None)
            thread_activity.release_delete(thread_id)

    @router.get("/artifacts/{artifact_id}")
    async def get_artifact(
        artifact_id: str,
        user: Principal = Depends(principal),
    ) -> Response:
        """Return one validated generated file owned by the authenticated user."""

        record = get_artifact_store().get(str(user.id), artifact_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        metadata, content = record
        safe_name = metadata.filename.replace('"', "")
        disposition = "inline" if metadata.kind == "image" else "attachment"
        return Response(
            content=content,
            media_type=metadata.mime_type,
            headers={
                "Content-Disposition": f'{disposition}; filename="{safe_name}"',
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return router
