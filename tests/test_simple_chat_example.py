"""The documented simple-chat example must remain executable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from juena_core.server.agent import runtime_model_middleware
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext

# The example is deliberately application code outside the installable
# ``src/juena_core`` package. Its documented entry points run from the
# repository root, so give this test the same import root pytest omits.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.simple_chat import agent as demo_agent
from examples.simple_chat.config import build_settings


def test_demo_settings_are_local_only() -> None:
    settings = build_settings({})

    assert settings.BIND_HOST == "127.0.0.1"
    assert settings.API_PUBLISHED is False
    assert settings.DEFAULT_PROVIDER == "openai"
    assert settings.DATABASE_URL is not None


def test_demo_rejects_an_unknown_provider() -> None:
    with pytest.raises(RuntimeError, match="DEMO_PROVIDER must be one of"):
        build_settings({"DEMO_PROVIDER": "unknown"})


@pytest.mark.asyncio
async def test_demo_agent_factory_compiles_without_a_model_call(monkeypatch) -> None:
    fake_model = FakeMessagesListChatModel(responses=[AIMessage(content="Hello")])
    monkeypatch.setattr(demo_agent, "build_chat_model", lambda **_kwargs: fake_model)
    monkeypatch.setattr(demo_agent, "get_checkpointer", InMemorySaver)
    monkeypatch.setattr(
        runtime_model_middleware,
        "build_chat_model",
        lambda **_kwargs: fake_model,
    )

    resources, graph = await demo_agent.build_demo_agent("openai", "fake-model")
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Hi")]},
        config={"configurable": {"thread_id": "demo-test"}},
        context=RuntimeModelContext(provider="openai", model="fake-model"),
    )

    assert resources is not None
    assert graph.name == demo_agent.DEMO_AGENT_ID
    assert isinstance(graph.checkpointer, InMemorySaver)
    assert result["messages"][-1].content == "Hello"
