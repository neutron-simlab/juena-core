"""Authenticated controls exercised concurrently with a blocked HTTP invocation."""

import asyncio
import json
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI, Request
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from juena_core.agents.steering import SteeringMiddleware
from juena_core.artifacts import ArtifactStore, set_artifact_store_for_tests
from juena_core.server.api import endpoints
from juena_core.server.chat.repository import ChatNotFoundError
from juena_core.server.database.connection import get_db_session
from juena_core.server.run_control import RunRegistry
from test_steering import ScriptedModel

OWNER = UUID("00000000-0000-0000-0000-000000000001")


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint_failure", [False, True])
async def test_controls_during_blocked_stream_are_authenticated_and_do_not_start_another_run(
    configured, monkeypatch, tmp_path, checkpoint_failure
):
    entered, release = asyncio.Event(), asyncio.Event()
    observed = []

    @tool
    async def search() -> str:
        """Wait for a fixture response."""
        entered.set()
        await release.wait()
        return "evidence"

    class Inspect(SteeringMiddleware):
        async def awrap_model_call(self, request, handler):
            observed.append([m.content for m in request.messages])
            return await super().awrap_model_call(request, handler)

    graph = create_agent(
        ScriptedModel(
            responses=[
                AIMessage(
                    content="", tool_calls=[{"name": "search", "args": {}, "id": "s"}]
                ),
                AIMessage(content="answer"),
            ]
        ),
        tools=[search],
        middleware=[Inspect("supervisor")],
        checkpointer=InMemorySaver(),
    )
    registry = RunRegistry()
    monkeypatch.setattr(endpoints, "RunRegistry", lambda: registry)

    async def get_agent(*args, **kwargs):
        return graph

    async def ensure(session, user_id, thread_id, agent_id, **kwargs):
        if user_id != OWNER:
            raise ChatNotFoundError
        return SimpleNamespace(thread_id=thread_id, updated_at=None)

    async def owned(session, user_id, thread_id, *, agent_id):
        return await ensure(session, user_id, thread_id, agent_id)

    async def prepare(*args, **kwargs):
        return None

    class Session:
        async def commit(self):
            pass

    async def session():
        yield Session()

    async def principal(request: Request):
        return SimpleNamespace(id=UUID(request.headers.get("x-user", str(OWNER))))

    monkeypatch.setattr(endpoints, "get_agent", get_agent)
    monkeypatch.setattr(endpoints, "ensure_owned_chat", ensure)
    monkeypatch.setattr(endpoints, "get_owned_chat", owned)
    monkeypatch.setattr(endpoints, "prepare_code_chat_turn_inputs", prepare)
    set_artifact_store_for_tests(
        ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    )
    app = FastAPI()
    app.dependency_overrides[get_db_session] = session
    app.include_router(endpoints.build_api_router(principal, run_controls=True))
    if checkpoint_failure:

        async def fail_save(*args):
            raise RuntimeError("temporary checkpoint failure")

        monkeypatch.setattr("juena_core.server.run_control._save_delivery", fail_save)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fixture"
        ) as client:
            first = asyncio.create_task(
                client.post(
                    "/juena/stream", json={"thread_id": "thread", "message": "original"}
                )
            )
            await asyncio.wait_for(entered.wait(), 3)
            run = next(iter(registry.runs.values()))
            base = f"/threads/thread/runs/{run.id}"
            assert (
                await client.post(
                    "/juena/stream", json={"thread_id": "thread", "message": "second"}
                )
            ).status_code == 409
            foreign = {"x-user": "00000000-0000-0000-0000-000000000002"}
            assert (await client.get(base, headers=foreign)).status_code == 404
            assert (
                await client.post(base + "/stop", headers=foreign)
            ).status_code == 404
            assert (
                await client.post(
                    base + "/steer",
                    headers=foreign,
                    json={"instruction_id": "one", "text": "correction"},
                )
            ).status_code == 404
            assert (
                await client.post(
                    base + "/steer",
                    json={"instruction_id": "one", "text": "correction"},
                )
            ).status_code == 202
            assert (
                await client.post(
                    base + "/steer",
                    json={"instruction_id": "one", "text": "correction"},
                )
            ).status_code == 202
            assert (
                await client.post(
                    base + "/steer", json={"instruction_id": "one", "text": "different"}
                )
            ).status_code == 409
            assert (
                await client.post(
                    base + "/steer", json={"instruction_id": "blank", "text": "  "}
                )
            ).status_code == 422
            assert (await client.get(base)).json()["instructions"][0][
                "status"
            ] == "received"
            release.set()
            response = await asyncio.wait_for(first, 3)
            assert response.status_code == 200
            assert "correction" in observed[-1]
            status = (await client.get(base)).json()
            assert status["state"] == ("failed" if checkpoint_failure else "completed")
            assert status["instructions"][0]["status"] == "applied"
            frames = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            assert frames[0] == {"type": "thread", "thread_id": "thread"}
            assert frames[1]["type"] == "run"
            assert (
                await client.post(
                    base + "/steer", json={"instruction_id": "late", "text": "too late"}
                )
            ).status_code == 409
            assert (
                await client.post(base.replace(run.id, "stale") + "/stop")
            ).status_code == 404
            # A fresh invocation can start after completion.
            fresh = await client.post(
                "/juena/stream", json={"thread_id": "thread", "message": "fresh"}
            )
            assert fresh.status_code == 200
    finally:
        set_artifact_store_for_tests(None)
