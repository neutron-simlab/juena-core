"""Agent middleware for approved commands and typed execution evidence."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from juena_core.agents.specialist_outcome import SpecialistOutcomeState
from juena_core.artifacts import get_artifact_store
from juena_core.runtime_context import _context_value
from juena_core.sandbox.evidence import capture_sandbox_execution
from juena_core.sandbox.policy import requires_execution_approval
from juena_core.schema.interrupts import ExecutionEvidence

__all__ = ["SandboxExecutionMiddleware"]


def _identity(runtime: Any) -> tuple[str, str, str | None] | None:
    """Return ``(user_id, thread_id, run_id)`` for an authenticated call."""

    context = getattr(runtime, "context", None)
    user_id = _context_value(context, "user_id")
    thread_id = _context_value(context, "thread_id")
    if not user_id or not thread_id:
        return None
    config = getattr(runtime, "config", None)
    run_id = config.get("run_id") if hasattr(config, "get") else None
    return user_id, thread_id, str(run_id) if run_id else None


def _graph_run_id(runtime: Any) -> str:
    """Return the invocation ID used to scope checkpointed evidence.

    The runtime *context* is asked first, and it is the only source that works
    where this middleware actually runs. `execute` is bound inside a specialist
    subagent, and LangGraph fills `execution_info.run_id` from the config of
    the graph that is running -- a subgraph does not inherit the parent's, so
    there it is `None`. The context is passed down unchanged, so it is the one
    identity a specialist and the supervisor reading its evidence can agree on.
    The other two are kept for a caller that invokes an agent directly, without
    going through the server's run context.
    """

    value = _context_value(getattr(runtime, "context", None), "run_id")
    if value:
        return value
    execution_info = getattr(runtime, "execution_info", None)
    value = getattr(execution_info, "run_id", None)
    if value:
        return str(value)
    config = getattr(runtime, "config", None)
    value = config.get("run_id") if hasattr(config, "get") else None
    if value:
        return str(value)
    raise RuntimeError("Sandbox execution requires a graph run_id")


def _tool_message(result: Any) -> ToolMessage | None:
    if isinstance(result, ToolMessage):
        return result
    update = getattr(result, "update", None)
    messages = update.get("messages", []) if isinstance(update, dict) else []
    for message in reversed(messages if isinstance(messages, list) else []):
        if isinstance(message, ToolMessage):
            return message
    return None


def _result_text(result: Any) -> str:
    message = _tool_message(result)
    return message.text if message is not None else ""


class SandboxExecutionMiddleware(AgentMiddleware[SpecialistOutcomeState]):
    """Capture execution facts and audit commands that passed human review."""

    state_schema = SpecialistOutcomeState

    @staticmethod
    def _command(request: Any) -> str | None:
        tool_call = request.tool_call
        if not isinstance(tool_call, dict) or tool_call.get("name") != "execute":
            return None
        args = tool_call.get("args")
        command = args.get("command") if isinstance(args, dict) else None
        if not isinstance(command, str) or not command.strip():
            return None
        return command

    @staticmethod
    def _before(
        request: Any,
        command: str,
    ) -> tuple[str, str, str | None, str] | None:
        """Persist the command about to run and return its audit identity."""

        if not requires_execution_approval(request):
            return None
        identity = _identity(getattr(request, "runtime", None))
        if identity is None:
            return None
        user_id, thread_id, run_id = identity
        try:
            get_artifact_store().register_artifact(
                user_id=user_id,
                thread_id=thread_id,
                run_id=run_id,
                filename="generated-command.sh",
                content=(command.rstrip() + "\n").encode("utf-8"),
                caption="Generated command",
                category="record",
            )
        except (ValueError, OSError):
            pass
        return user_id, thread_id, run_id, command

    @staticmethod
    def _after(record: tuple[str, str, str | None, str], result: Any) -> None:
        user_id, thread_id, run_id, command = record
        text = _result_text(result)
        store = get_artifact_store()
        if text:
            try:
                store.register_artifact(
                    user_id=user_id,
                    thread_id=thread_id,
                    run_id=run_id,
                    filename="execution-output.txt",
                    content=text.encode("utf-8"),
                    caption="Execution output",
                    category="record",
                )
            except (ValueError, OSError):
                pass
        store.audit(
            {
                "event": "sandbox_execution_approved",
                "user_id": user_id,
                "thread_id": thread_id,
                "run_id": run_id,
                "command": command,
            }
        )

    @staticmethod
    def _with_evidence(
        result: Any,
        command: str,
        graph_run_id: str,
        events: list[ExecutionEvidence],
    ) -> Command:
        message = _tool_message(result)
        if not events:
            events = [
                ExecutionEvidence(
                    graph_run_id=graph_run_id,
                    command=command,
                    status="tool_error",
                    exit_code=None,
                )
            ]
        last = events[-1]
        if message is not None and last.attempted:
            message.status = "success" if last.succeeded else "error"
        update = getattr(result, "update", None)
        merged = dict(update) if isinstance(update, dict) else {}
        if isinstance(result, ToolMessage):
            merged["messages"] = [result]
        merged["execution_events"] = [
            event.model_dump(mode="json") for event in events
        ]
        if isinstance(result, Command):
            return replace(result, update=merged)
        return Command(update=merged)

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        command = self._command(request)
        if command is None:
            return handler(request)
        runtime = getattr(request, "runtime", None)
        graph_run_id = _graph_run_id(runtime)
        record = self._before(request, command)
        with capture_sandbox_execution(graph_run_id) as events:
            result = handler(request)
        if record is not None:
            self._after(record, result)
        return self._with_evidence(result, command, graph_run_id, events)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        command = self._command(request)
        if command is None:
            return await handler(request)
        runtime = getattr(request, "runtime", None)
        graph_run_id = _graph_run_id(runtime)
        record = self._before(request, command)
        with capture_sandbox_execution(graph_run_id) as events:
            result = await handler(request)
        if record is not None:
            self._after(record, result)
        return self._with_evidence(result, command, graph_run_id, events)
