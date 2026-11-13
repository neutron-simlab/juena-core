"""What crosses into and out of a specialist when the supervisor delegates to it.

`SubAgentMiddleware` hands a subagent the parent's whole state minus its private
keys, and copies the subagent's whole state back the same way. Here that is
wrong in both directions:

*Going in.* The supervisor's `files` channel is not only findings.
Summarization saves removed conversation turns under `/conversation_history/`
in the same channel, and a specialist, which is told it cannot see the
conversation, must not receive them.

*Coming back.* Everything the run left in `files` returns, including findings it
only inherited on the way in. Rewriting an unchanged file undoes any edit made
to it in the meantime.

So each specialist is registered through this adapter, which names what may
cross in each direction. A state field added by a later library release then
stays out until someone lets it cross.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig

from juena_core.agents.findings import crossing_files, findings_delta

__all__ = [
    "USER_MESSAGE_NOTE",
    "AttachUserMessageMiddleware",
    "SpecialistDelegate",
    "with_delegation_boundary",
    "with_user_message",
]

#: Tells the specialist what the attachment is for. The objective stays the
#: assignment: when the supervisor splits one request between two specialists,
#: each must not take on the whole message.
USER_MESSAGE_NOTE = (
    "The user's message this turn, attached automatically by the server. Your "
    "assignment is the objective above; use this message only for details that "
    "objective refers to, such as a specification it mentions."
)


def with_user_message(description: str, messages: Sequence[Any]) -> str:
    """Return the brief with the user's latest message attached.

    Only the latest message: an earlier one may hold a request the user has since
    replaced, so carrying context from earlier turns stays the supervisor's job.
    Nothing is added when the brief already contains the message word for word.

    Args:
        description: The brief the supervisor wrote for the specialist.
        messages: The supervisor's message history.

    Returns:
        The brief, with the user's latest message attached when needed.
    """

    # Corrections have their own delivery path into each specialist. Attaching
    # one here would both duplicate it and drop the original specification.
    user = next((
        m for m in reversed(messages)
        if isinstance(m, HumanMessage) and "steering" not in m.additional_kwargs
    ), None)
    text = user.text.strip() if user is not None else ""
    if not text or text in description:
        return description
    return f"{description}\n\n---\n{USER_MESSAGE_NOTE}\n\n<user_message>\n{text}\n</user_message>"


class AttachUserMessageMiddleware(AgentMiddleware):
    """Attach the user's latest message to every brief the supervisor writes.

    A specialist sees only its brief, never the conversation (see the module
    docstring). A supervisor model sometimes writes "per the user's spec" and
    leaves the spec out; the specialist then cannot do the task and has to ask.
    Asking the supervisor more firmly did not stop that, so the server attaches
    the message itself.

    It changes only what the `task` tool receives. The supervisor's own message
    history keeps the brief as the model wrote it.
    """

    def _attached(self, request: Any) -> Any:
        call = request.tool_call
        args = call.get("args") or {}
        description = args.get("description")
        if call.get("name") != "task" or not isinstance(description, str):
            return request
        brief = with_user_message(description, request.state.get("messages", []))
        if brief == description:
            return request
        return request.override(tool_call={**call, "args": {**args, "description": brief}})

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Run a tool call, with the user's message attached to a `task` brief.

        Args:
            request: The tool-call request.
            handler: Runs the request.

        Returns:
            The handler's result.
        """
        return handler(self._attached(request))

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Async version of :meth:`wrap_tool_call`.

        Args:
            request: The tool-call request.
            handler: Runs the request.

        Returns:
            The handler's result.
        """
        return await handler(self._attached(request))


class SpecialistDelegate(Runnable[dict[str, Any], dict[str, Any]]):
    """One specialist, invoked with an allowlist rather than a snapshot."""

    def __init__(self, runnable: Runnable, *, name: str) -> None:
        """Initialize the wrapper.

        Args:
            runnable: The compiled specialist.
            name: The specialist's name.
        """
        self._runnable = runnable
        self.name = name

    def _inbound(self, state: Any) -> dict[str, Any]:
        """Exactly the objective and the files the contract names.

        Built from scratch rather than filtered, so a state field added upstream
        is absent by default instead of crossing until someone notices.
        """
        values = state if isinstance(state, dict) else {}
        return {
            "messages": values.get("messages") or [],
            "files": crossing_files(values.get("files")),
        }

    def _outbound(self, sent: dict[str, Any], result: Any) -> dict[str, Any]:
        """The report, and the findings this run actually wrote."""
        values = result if isinstance(result, dict) else {}
        return {
            "messages": values.get("messages") or [],
            "files": findings_delta(sent["files"], values.get("files")),
        }

    def invoke(
        self,
        input: dict[str, Any],  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the specialist on the allowed part of the supervisor's state.

        Args:
            input: The supervisor's state.
            config: Run config passed to the specialist.
            **kwargs: Passed to the specialist's ``invoke``.

        Returns:
            The specialist's messages and the findings this run wrote.
        """
        sent = self._inbound(input)
        return self._outbound(sent, self._runnable.invoke(sent, config, **kwargs))

    async def ainvoke(
        self,
        input: dict[str, Any],  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of :meth:`invoke`.

        Args:
            input: The supervisor's state.
            config: Run config passed to the specialist.
            **kwargs: Passed to the specialist's ``ainvoke``.

        Returns:
            The specialist's messages and the findings this run wrote.
        """
        sent = self._inbound(input)
        return self._outbound(sent, await self._runnable.ainvoke(sent, config, **kwargs))


def with_delegation_boundary(specialists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap each compiled specialist for registration with `SubAgentMiddleware`.

    A specialist that is already a `SpecialistDelegate` is left alone. An
    application may subclass it to return one extra field, and
    `build_supervisor_middleware` still passes every specialist through here.
    Wrapping again would put core's outbound allowlist outside the
    application's and silently drop that field. A second wrapper can only ever
    remove what the first allowed.

    Args:
        specialists: Specialist specs, each with a ``name`` and a ``runnable``.

    Returns:
        The specs, each runnable wrapped in a `SpecialistDelegate`.
    """

    return [
        spec
        if isinstance(spec["runnable"], SpecialistDelegate)
        else {**spec, "runnable": SpecialistDelegate(spec["runnable"], name=spec["name"])}
        for spec in specialists
    ]
