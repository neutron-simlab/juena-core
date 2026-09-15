"""The demo application's only agent."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from juena_core.llms_providers import build_chat_model
from juena_core.server.agent.runtime_model_middleware import (
    RuntimeModelContext,
    RuntimeModelMiddleware,
)
from juena_core.server.database.checkpointer import get_checkpointer

DEMO_AGENT_ID = "demo"

SYSTEM_PROMPT = """You are a concise and friendly demonstration assistant.
Answer the user's question directly. Say when you are uncertain. You have no
external tools, so never claim to have searched the web or changed a file.
"""


async def build_demo_agent(
    provider: str,
    model: str,
) -> tuple[Any, CompiledStateGraph]:
    """Build the graph on its first request, inside the server lifespan."""

    default_model = build_chat_model(
        provider=provider,
        model=model,
        temperature=0.2,
        streaming=True,
    )
    graph = create_agent(
        model=default_model,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        middleware=[RuntimeModelMiddleware()],
        context_schema=RuntimeModelContext,
        checkpointer=get_checkpointer(),
        name=DEMO_AGENT_ID,
    )
    # The first value is where an application can retain closeable clients or
    # other process-lifetime resources. This demo owns none.
    return object(), graph
