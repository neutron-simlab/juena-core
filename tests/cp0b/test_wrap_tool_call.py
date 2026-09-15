"""CP0b, rows 4 and 5: a ``@wrap_tool_call`` middleware reads ``request.runtime``
(state, thread_id via the runtime's context/config), and returning
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

A third finding: ``request.runtime.context`` is ``None`` unless a
``context_schema``/``context=`` is used. The thread id a façade tool needs
(03/CP3a) is not there — it is in
``request.runtime.config["configurable"]["thread_id"]``, the same place
LangGraph checkpointing already reads it from.
"""

from typing import Annotated, NotRequired

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, wrap_tool_call
from langchain.mcp import MCPAdapter
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _keep_last(_a, b):
    return b


class _ExtraState(AgentState):
    captured: NotRequired[Annotated[bool, _keep_last]]


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

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "call_1", "type": "tool_call"}],
    )
    model = _ScriptedModel(messages=iter([tool_call_msg, AIMessage(content="done")]))

    agent = create_agent(model=model, tools=tools, middleware=[capture])
    result = await agent.ainvoke(
        {"messages": [HumanMessage("go")]},
        {"configurable": {"thread_id": "cp0b-wrap-tool-call"}},
    )

    # request.runtime carries the pieces a façade tool needs -- state and,
    # via config (not context), the thread id.
    runtime = seen["runtime"]
    assert "messages" in seen["state"]
    assert runtime.config["configurable"]["thread_id"] == "cp0b-wrap-tool-call"

    # Command(update=...) reached the declared state channel.
    assert result["captured"] is True
