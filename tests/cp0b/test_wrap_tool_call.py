"""CP0b, rows 4 and 5: a ``@wrap_tool_call`` middleware reads ``request.runtime``
(state, typed context, and thread_id via execution info), and returning
``Command(update=...)`` from it writes state *and* the tool message.

One thing this found that the plan text did not originally say: the
decorated function must be **async** (``await handler(request)``) to run
under ``agent.ainvoke``/``astream`` — a sync ``@wrap_tool_call`` raises
``NotImplementedError`` there. juena_core's server is async throughout
(SSE streaming), so every wrap_tool_call middleware it defines must be async.

A second finding: ``Command(update={...})`` only persists keys that exist in
a declared state schema. An undeclared key is silently dropped, not an
error — this is exactly why 00's `execution_events` channel and 02/CP2's
`captured`-style fields must be declared via ``state_schema=`` (or the
agent's own schema), not assumed to "just work" because the middleware wrote
them.

A third finding: ``request.runtime.context`` is ``None`` when no
``context_schema``/``context=`` is used, and carries the supplied typed context
when both are present. The public thread identity a façade tool needs (03/CP3a)
is ``request.runtime.execution_info.thread_id``. The underlying
``request.runtime.config["configurable"]["thread_id"]`` carries the matching
checkpoint key, but application code does not need to reach into it.
"""

from dataclasses import dataclass
from typing import Annotated, NotRequired

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, wrap_tool_call
from langchain.mcp import MCPAdapter
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _keep_last(_a, b):
    return b


class _ExtraState(AgentState):
    captured: NotRequired[Annotated[bool, _keep_last]]


@dataclass(frozen=True)
class _RuntimeContext:
    principal_id: str


def _scripted_model() -> _ScriptedModel:
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "add", "args": {"a": 2, "b": 3}, "id": "call_1", "type": "tool_call"}
        ],
    )
    return _ScriptedModel(messages=iter([tool_call_msg, AIMessage(content="done")]))


@pytest.mark.asyncio
async def test_middleware_sees_runtime_and_command_writes_state(scratch_mcp_server):
    seen = {}

    @wrap_tool_call(state_schema=_ExtraState)
    async def capture(request, handler):
        seen["runtime"] = request.runtime
        seen["state"] = request.state
        result = await handler(request)
        return Command(update={"messages": [result], "captured": True})

    async with MCPAdapter(scratch_mcp_server) as adapter:
        tools = await adapter.list_tools()

    context = _RuntimeContext(principal_id="principal-1")
    agent = create_agent(
        model=_scripted_model(),
        tools=tools,
        middleware=[capture],
        context_schema=_RuntimeContext,
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("go")]},
        {"configurable": {"thread_id": "cp0b-wrap-tool-call"}},
        context=context,
    )

    # Typed context and execution identity are separate runtime channels.
    runtime = seen["runtime"]
    assert "messages" in seen["state"]
    assert runtime.context == context
    assert runtime.execution_info.thread_id == "cp0b-wrap-tool-call"
    # This is the underlying checkpoint key, not the façade's preferred API.
    assert runtime.config["configurable"]["thread_id"] == "cp0b-wrap-tool-call"

    # Command(update=...) preserved the tool response and reached the declared channel.
    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_1"
    assert tool_messages[0].artifact["structured_content"] == {"total": 5}
    assert result["captured"] is True


@pytest.mark.asyncio
async def test_command_drops_undeclared_state_key(scratch_mcp_server):
    @wrap_tool_call
    async def write_undeclared(request, handler):
        result = await handler(request)
        return Command(update={"messages": [result], "undeclared": True})

    async with MCPAdapter(scratch_mcp_server) as adapter:
        tools = await adapter.list_tools()

    agent = create_agent(model=_scripted_model(), tools=tools, middleware=[write_undeclared])
    result = await agent.ainvoke({"messages": [HumanMessage("go")]})

    assert "undeclared" not in result
    assert any(isinstance(message, ToolMessage) for message in result["messages"])


@pytest.mark.asyncio
async def test_sync_wrap_tool_call_is_rejected_by_async_agent(scratch_mcp_server):
    @wrap_tool_call
    def sync_wrapper(request, handler):
        return handler(request)

    async with MCPAdapter(scratch_mcp_server) as adapter:
        tools = await adapter.list_tools()

    agent = create_agent(model=_scripted_model(), tools=tools, middleware=[sync_wrapper])

    with pytest.raises(NotImplementedError):
        await agent.ainvoke({"messages": [HumanMessage("go")]})
