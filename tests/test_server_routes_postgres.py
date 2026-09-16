"""CP4's routes, through a real application against a real Postgres.

Requires the throwaway database:
``docker compose -f tests/compose.postgres.yml up -d --wait``

These do not skip when Postgres is unreachable. The properties here are about
what the running application does — whether the lifespan wrote a row before
serving, whether a route refuses a thread belonging to another agent — and a
skip would turn "not measured" into "passed".

The agent registered here is a two-node graph, not a model. Every property
under test is decided before the graph is invoked, or by the checkpointer after
it is; none of it depends on what a language model would say.
"""

from __future__ import annotations

from pathlib import Path
from datetime import timedelta
from typing import TypedDict
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph

from juena_core import config as config_module
from juena_core.config import CoreSettings
from juena_core.schema.llm_models import BlabladorModelName
from juena_core.server.agent import registry as registry_module
from juena_core.server.database import connection as connection_module
from juena_core.server.database.checkpointer import get_checkpointer
from juena_core.server.database.models import Base, Chat, User, utc_now
from juena_core.server.identity import local_principal
from juena_core.server.service import create_app

DSN = "postgresql://juena_core_test:juena_core_test@127.0.0.1:55432/juena_core_test"

SIMULATOR = "simulator"
ADVANCED = "advanced_mode"


def make_settings(**overrides: object) -> CoreSettings:
    values: dict[str, object] = {
        "OPENAI_API_KEY": None,
        "BLABLADOR_API_KEY": None,
        "BLABLADOR_BASE_URL": None,
        "MAX_TOKENS": 10_000,
        "TIMEOUT_SECONDS": 60,
        "MAX_RETRIES": 3,
        "DEFAULT_PROVIDER": "blablador",
        "DEFAULT_MODEL": BlabladorModelName.GPT_OSS.value,
        "OPENAI_AVAILABLE_MODELS": None,
        "BLABLADOR_AVAILABLE_MODELS": None,
        "OPENAI_DEFAULT_MODEL": "gpt-4o-mini",
        "BLABLADOR_DEFAULT_MODEL": BlabladorModelName.GPT_OSS.value,
        "STREAM_TOOL_PAYLOADS": False,
        "DATABASE_URL": DSN,
        "DATABASE_POOL_MAX_SIZE": 5,
        "SESSION_TTL_HOURS": 8,
        "SESSION_COOKIE_SECURE": False,
        "FALLBACK_PROVIDER": None,
        "EXECUTE_TIMEOUT_SECONDS": 600,
        "ARTIFACT_ROOT": Path("/tmp/juena-core-route-artifacts"),
        "AUDIT_FILE": Path("/tmp/juena-core-route-artifacts/audit.jsonl"),
        "LOG_LEVEL": "INFO",
        "LOG_DIR": Path("/tmp/juena-core-route-logs"),
        "BIND_HOST": "127.0.0.1",
        "API_PUBLISHED": False,
    }
    values.update(overrides)
    return CoreSettings(**values)  # type: ignore[arg-type]


class _State(TypedDict):
    messages: list


def _noop(state: _State) -> dict:
    return {}


async def _agent_factory(provider: str, model: str):
    """Compile against the live checkpointer, as a real factory does.

    A factory runs on the first request, inside the lifespan, so
    ``get_checkpointer()`` is available — and without it ``aget_state`` raises
    "No checkpointer set", which is what a pending-interrupt lookup calls.
    """

    graph = StateGraph(_State)
    graph.add_node("noop", _noop)
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)
    return object(), graph.compile(checkpointer=get_checkpointer())


@pytest.fixture
def user_id() -> UUID:
    return uuid4()


@pytest.fixture
def client(monkeypatch, user_id):
    """A running application, with the schema rebuilt from nothing."""

    monkeypatch.setattr(config_module, "_settings", make_settings())
    monkeypatch.setattr(registry_module, "_agent_registry", {})
    monkeypatch.setattr(registry_module, "_agent_factories", {})
    monkeypatch.setattr(registry_module, "DEFAULT_AGENT", None)
    registry_module.register_agent_factory(SIMULATOR, _agent_factory, set_as_default=True)
    registry_module.register_agent_factory(ADVANCED, _agent_factory)

    app = create_app(principal=local_principal(user_id=user_id), title="test-app")
    with TestClient(app) as running:
        yield running


@pytest.fixture(autouse=True)
def _empty_schema():
    """Drop and rebuild the tables around each test, synchronously."""

    import psycopg

    def reset() -> None:
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute("DROP SCHEMA public CASCADE")
            connection.execute("CREATE SCHEMA public")

    reset()
    yield
    reset()


def _create_chat(client: TestClient, agent_id: str) -> str:
    response = client.post("/chats", json={"agent_id": agent_id, "title": "T"})
    assert response.status_code == 201, response.text
    return response.json()["thread_id"]


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------


def test_the_lifespan_creates_the_fixed_principals_row_before_serving(client, user_id) -> None:
    """Without this the first conversation fails on ``chats.user_id``.

    The chat is created through the real route with no manual call to
    ``ensure_principal_row``, so a lifespan that skipped it would show up here
    as a foreign-key violation rather than as a passing test.
    """

    thread_id = _create_chat(client, SIMULATOR)

    factory = connection_module.get_session_factory()

    async def stored() -> User | None:
        async with factory() as session:
            return await session.get(User, user_id)

    import asyncio

    assert asyncio.run(stored()) is not None
    assert client.get(f"/chats/{thread_id}").status_code == 200


def test_health_reports_the_application_not_the_package(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["details"]["service"] == "test-app"


# --------------------------------------------------------------------------
# A thread belongs to one agent
# --------------------------------------------------------------------------


def test_a_chat_records_the_agent_that_created_it(client) -> None:
    thread_id = _create_chat(client, SIMULATOR)
    assert client.get(f"/chats/{thread_id}").json()["agent_id"] == SIMULATOR


def test_listing_can_be_narrowed_to_one_agent(client) -> None:
    simulator_thread = _create_chat(client, SIMULATOR)
    advanced_thread = _create_chat(client, ADVANCED)

    everything = {chat["thread_id"] for chat in client.get("/chats").json()}
    narrowed = [chat["thread_id"] for chat in client.get(f"/chats?agent_id={ADVANCED}").json()]

    assert everything == {simulator_thread, advanced_thread}
    assert narrowed == [advanced_thread]


def test_authorizing_a_message_touches_chat_recency(client) -> None:
    """A new turn makes its conversation the most recent one in the list."""

    older = _create_chat(client, SIMULATOR)
    newer = _create_chat(client, SIMULATOR)
    factory = connection_module.get_session_factory()

    async def age_chats() -> None:
        async with factory() as session:
            older_chat = await session.get(Chat, older)
            newer_chat = await session.get(Chat, newer)
            assert older_chat is not None and newer_chat is not None
            older_chat.updated_at = utc_now() - timedelta(days=2)
            newer_chat.updated_at = utc_now() - timedelta(days=1)
            await session.commit()

    import asyncio

    asyncio.run(age_chats())
    assert client.get(f"/chats?agent_id={SIMULATOR}").json()[0]["thread_id"] == newer

    response = client.post(
        f"/{SIMULATOR}/stream",
        json={"message": "touch this thread", "thread_id": older},
    )
    assert response.status_code == 200
    assert client.get(f"/chats?agent_id={SIMULATOR}").json()[0]["thread_id"] == older


def test_streaming_another_agents_thread_is_not_found(client) -> None:
    thread_id = _create_chat(client, SIMULATOR)
    response = client.post(f"/{ADVANCED}/stream", json={"message": "hi", "thread_id": thread_id})
    assert response.status_code == 404
    assert response.json()["detail"] == "Chat not found"


def test_streaming_with_files_under_another_agent_is_not_found(client) -> None:
    thread_id = _create_chat(client, SIMULATOR)
    response = client.post(
        f"/{ADVANCED}/stream_with_files",
        data={"message": "hi", "thread_id": thread_id},
    )
    assert response.status_code == 404


def test_resuming_under_another_agent_is_not_found(client) -> None:
    thread_id = _create_chat(client, SIMULATOR)
    response = client.post(
        f"/{ADVANCED}/resume",
        json={
            "kind": "clarification",
            "thread_id": thread_id,
            "interrupt_id": "i-1",
            "answer": "yes",
        },
    )
    assert response.status_code == 404


def test_a_pending_interrupt_lookup_under_another_agent_is_not_found(client) -> None:
    """So a card raised by one agent cannot be answered into the other."""

    thread_id = _create_chat(client, SIMULATOR)
    response = client.get(f"/threads/{thread_id}/pending-interrupt?agent_id={ADVANCED}")
    assert response.status_code == 404


def test_the_owning_agent_reaches_its_own_thread(client) -> None:
    """The mirror of the four refusals: the checks reject a mismatch, not
    everything."""

    thread_id = _create_chat(client, SIMULATOR)
    response = client.get(f"/threads/{thread_id}/pending-interrupt?agent_id={SIMULATOR}")
    assert response.status_code == 200
    assert response.json() == {"approval": None, "clarification": None}


def test_reusing_a_thread_id_under_another_agent_is_not_found(client) -> None:
    """Ownership is checked on creation too, not only on lookup."""

    thread_id = _create_chat(client, SIMULATOR)
    response = client.post("/chats", json={"agent_id": ADVANCED, "thread_id": thread_id})
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Deletion
# --------------------------------------------------------------------------


def test_a_thread_is_deleted_whichever_agent_owns_it(client) -> None:
    """The path carries no agent, and a caller may delete a conversation of
    theirs under any of them."""

    thread_id = _create_chat(client, ADVANCED)
    assert client.delete(f"/threads/{thread_id}").status_code == 200
    assert client.get(f"/chats/{thread_id}").status_code == 404


def test_deleting_an_unknown_thread_is_not_found(client) -> None:
    assert client.delete(f"/threads/{uuid4()}").status_code == 404


def test_the_workspace_hook_runs_on_delete(monkeypatch, user_id) -> None:
    """An application that materialises files outside graph state gets told to
    clean them up."""

    from juena_core.server.api.endpoints import ThreadWorkspace

    monkeypatch.setattr(config_module, "_settings", make_settings())
    monkeypatch.setattr(registry_module, "_agent_registry", {})
    monkeypatch.setattr(registry_module, "_agent_factories", {})
    monkeypatch.setattr(registry_module, "DEFAULT_AGENT", None)
    registry_module.register_agent_factory(SIMULATOR, _agent_factory, set_as_default=True)

    deleted: list[tuple[str, str]] = []

    async def remove(*, user_id: str, thread_id: str) -> None:
        deleted.append((user_id, thread_id))

    app = create_app(
        principal=local_principal(user_id=user_id),
        workspace=ThreadWorkspace(delete=remove),
    )
    with TestClient(app) as client:
        thread_id = _create_chat(client, SIMULATOR)
        assert client.delete(f"/threads/{thread_id}").status_code == 200

    assert deleted == [(str(user_id), thread_id)]


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------


def test_an_unknown_artifact_is_not_found(client) -> None:
    assert client.get(f"/artifacts/{uuid4()}").status_code == 404
