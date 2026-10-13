"""The tracing hook: what an application's tracer receives on every run.

No tracing vendor is installed here. These tests pin the contract a callback
tracer such as Langfuse relies on: its handler and metadata ride on the config
``build_run_context`` returns, and the root run of an invocation carries both.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda

from juena_core import tracing
from juena_core.server.agent.input_handler import AgentInputHandler
from juena_core.tracing import RunTracing, configure_tracing


class _Recorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.chain_starts: list[dict[str, Any]] = []

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self.chain_starts.append(
            {
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "metadata": kwargs.get("metadata") or {},
            }
        )


@pytest.fixture
def no_hook(monkeypatch):
    """Start every test without a hook and restore whatever was there."""

    monkeypatch.setattr(tracing, "_hook", None)


def test_without_a_hook_the_run_config_carries_no_tracing(configured, no_hook) -> None:
    run = AgentInputHandler.build_run_context(thread_id="thread-1", user_id="user-1")

    assert "callbacks" not in run.config
    assert "metadata" not in run.config


def test_hook_receives_the_run_identity_and_extends_the_config(
    configured, no_hook
) -> None:
    recorder = _Recorder()
    calls: list[dict[str, str]] = []

    def hook(*, thread_id: str, user_id: str) -> RunTracing:
        calls.append({"thread_id": thread_id, "user_id": user_id})
        return RunTracing(callbacks=(recorder,), metadata={"session": thread_id})

    configure_tracing(hook)
    # No thread id given: the hook must see the one core generated.
    run = AgentInputHandler.build_run_context(user_id="user-1")

    assert calls == [{"thread_id": run.thread_id, "user_id": "user-1"}]
    assert run.config["callbacks"] == [recorder]
    assert run.config["metadata"] == {"session": run.thread_id}
    assert run.config["configurable"]["thread_id"] == run.thread_id
    assert run.config["run_id"] == run.run_id


@pytest.mark.asyncio
async def test_root_run_carries_the_hook_metadata(configured, no_hook) -> None:
    recorder = _Recorder()
    configure_tracing(
        lambda *, thread_id, user_id: RunTracing(
            callbacks=(recorder,),
            metadata={"session": thread_id, "user": user_id},
        )
    )
    run = AgentInputHandler.build_run_context(thread_id="thread-1", user_id="user-1")

    await RunnableLambda(lambda value: value).ainvoke("hi", config=run.config)

    [root] = recorder.chain_starts
    assert root["run_id"] == run.run_id
    assert isinstance(root["run_id"], UUID)
    assert root["parent_run_id"] is None
    assert root["metadata"]["session"] == "thread-1"
    assert root["metadata"]["user"] == "user-1"


def test_configuring_the_same_hook_twice_is_harmless(no_hook) -> None:
    def hook(*, thread_id: str, user_id: str) -> RunTracing:
        return RunTracing()

    configure_tracing(hook)
    configure_tracing(hook)

    with pytest.raises(RuntimeError, match="already been called"):
        configure_tracing(lambda *, thread_id, user_id: RunTracing())
