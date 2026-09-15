"""CP0b, row 9: ``create_deep_agent(...)`` accepts the kwargs 03/CP5 passes
(``name``, ``model``, ``tools``, ``middleware``, ``subagents``, ``backend``,
``checkpointer``, ``system_prompt`` — read from
``vitess_ai/agents/advanced_mode/agent.py``), and a subagent dict's
``middleware`` key still runs during delegation.

This is a full run, not just a signature inspection: the top-level model is
scripted to call the ``task`` tool deepagents exposes for delegation, and the
test asserts the subagent's own middleware actually executed — a version
bump changing how subagents are wired would show up as the marker never
being set, not as an import error.
"""

import pytest
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@tool
def _noop_tool(x: str) -> str:
    """Echo x back."""
    return x


class _MarkerMiddleware(AgentMiddleware):
    def __init__(self, marker: dict):
        super().__init__()
        self._marker = marker

    async def awrap_tool_call(self, request, handler):
        self._marker["ran"] = True
        return await handler(request)


@pytest.mark.asyncio
async def test_subagent_middleware_runs_during_delegation():
    marker: dict = {}

    sub_model = _ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "_noop_tool", "args": {"x": "hi"}, "id": "c2", "type": "tool_call"}],
                ),
                AIMessage(content="sub done"),
            ]
        )
    )
    top_model = _ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {"description": "do it", "subagent_type": "sim-runner"},
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="top done"),
            ]
        )
    )

    agent = create_deep_agent(
        name="advanced_mode",
        model=top_model,
        tools=[],
        middleware=[],
        subagents=[
            {
                "name": "sim-runner",
                "description": "Runs sims",
                "system_prompt": "you run sims",
                "tools": [_noop_tool],
                "model": sub_model,
                "middleware": [_MarkerMiddleware(marker)],
            }
        ],
        backend=None,
        checkpointer=None,
        system_prompt="top prompt",
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("go")]},
        {"configurable": {"thread_id": "cp0b-deepagents"}},
    )

    assert marker.get("ran") is True
    assert result["messages"][-1].content == "top done"
