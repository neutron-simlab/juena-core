"""Tests for agent input preparation.

Moved here from juena-chatbot in plan 02/step 3: ``AgentInputHandler`` is
core's. The first test now takes the ``configured`` fixture from
``conftest.py``, because ``build_run_context`` resolves the provider and model
against ``settings()``, and core refuses to invent settings nobody configured.
"""

from __future__ import annotations

from uuid import UUID

from juena_core.server.agent.input_handler import AgentInputHandler
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext


def test_build_run_context_puts_identity_in_config_and_context(configured) -> None:
    run = AgentInputHandler.build_run_context(
        thread_id="thread-1",
        user_id="user-1",
        provider="openai",
        model="gpt-test",
    )

    assert isinstance(run.run_id, UUID)
    assert run.config["configurable"]["thread_id"] == "thread-1"
    assert run.config["configurable"]["user_id"] == "user-1"
    assert run.config["configurable"]["provider"] == "openai"
    assert run.config["configurable"]["model"] == "gpt-test"
    assert run.context == RuntimeModelContext(
        provider="openai",
        model="gpt-test",
        thread_id="thread-1",
        user_id="user-1",
        run_id=str(run.run_id),
    )
    # The same invocation id in both places, and that is the point of putting
    # it in the context at all: a subgraph does not inherit the config's
    # `run_id`, so a specialist writing execution evidence and the supervisor
    # reading it back would otherwise have no id in common.
    assert run.context.run_id == str(run.config["run_id"])


def test_build_input_carries_only_message_channels() -> None:
    input_data = AgentInputHandler.build_input("hello")

    assert input_data["messages"][0].content == "hello"
    # thread_id/user_id are not state channels: LangGraph drops unknown input
    # keys and logs a warning for each on every invocation. They travel in
    # config.configurable and the runtime context instead.
    assert set(input_data) == {"messages"}


def test_build_input_supports_message_override_and_initial_files() -> None:
    input_data = AgentInputHandler.build_input(
        "raw prompt",
        message_override="inspect /inputs first",
        initial_files={
            "/inputs/current_message.txt": {"content": ["raw prompt"], "created_at": "c", "modified_at": "m"},
            "/inputs/old.txt": None,
        },
    )

    assert input_data["messages"][0].content == "inspect /inputs first"
    assert input_data["files"] == {
        "/inputs/current_message.txt": {"content": ["raw prompt"], "created_at": "c", "modified_at": "m"},
        "/inputs/old.txt": None,
    }
