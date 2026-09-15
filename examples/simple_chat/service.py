"""FastAPI entry point for the simple chatbot example."""

from __future__ import annotations

from juena_core.config import configure
from juena_core.server.agent.registry import register_agent_factory
from juena_core.server.identity import local_principal
from juena_core.server.service import create_app

from examples.simple_chat.agent import DEMO_AGENT_ID, build_demo_agent
from examples.simple_chat.config import DEMO_USER_ID, build_settings

configure(build_settings())
register_agent_factory(DEMO_AGENT_ID, build_demo_agent, set_as_default=True)

app = create_app(
    principal=local_principal(
        user_id=DEMO_USER_ID,
        subject="simple-chat-demo",
        display_name="Local demo user",
    ),
    title="juena-core simple chat",
    version="0.1.0",
)
