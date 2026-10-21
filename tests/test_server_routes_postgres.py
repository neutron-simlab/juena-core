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
        "FALLBACK_PROVIDER": None,
        "EXECUTE_TIMEOUT_SECONDS": 600,
        "ARTIFACT_ROOT": Path("/tmp/juena-core-route-artifacts"),
        "AUDIT_FILE": Path("/tmp/juena-core-route-artifacts/audit.jsonl"),
        "LOG_LEVEL": "INFO",
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
def client(monkeypatch, user_id, request, tmp_path):
    """A running application, with the schema rebuilt from nothing."""

    monkeypatch.setattr(config_module, "_settings", make_settings())
    monkeypatch.setattr(registry_module, "_agent_registry", {})
    monkeypatch.setattr(registry_module, "_agent_factories", {})
    monkeypatch.setattr(registry_module, "DEFAULT_AGENT", None)
    registry_module.register_agent_factory(SIMULATOR, _agent_factory, set_as_default=True)
    registry_module.register_agent_factory(ADVANCED, _agent_factory)

    filesystem = None
    if getattr(request, "param", False):
        from juena_core.filesystem import service
        from juena_core.filesystem.storage import WorkspaceStore
        filesystem = service.ConversationFilesystem(WorkspaceStore(tmp_path / "workspaces"), identity_secret="postgres-tests-identity-secret-32-characters")
        monkeypatch.setattr(service, "_filesystem", filesystem)
        registry_module.register_agent_factory(SIMULATOR, _filesystem_agent_factory, set_as_default=True)
    app = create_app(principal=local_principal(user_id=user_id), title="test-app", filesystem=filesystem)
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


async def _filesystem_agent_factory(provider: str, model: str):
    from deepagents.middleware.filesystem import FilesystemState
    from langgraph.types import interrupt
    from juena_core.agents.specialist_runtime import build_specialist_backend
    from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext

    backend = build_specialist_backend(workspace_writable=True)

    async def observe(state):
        if "cancel" in state["messages"][-1].content:
            import asyncio
            await asyncio.sleep(60)
        if "pause" in state["messages"][-1].content:
            interrupt({"kind": "clarification", "question": "Continue?"})
        read = await backend.aread("/workspace/uploads/code.py")
        assert read.error is None, read.error
        # A nested working file appears without registration as a chat artifact.
        await backend.awrite("/workspace/nested/observed.py", read.file_data["content"])
        written = await backend.awrite("/findings/committed.md", "Specialist observation")
        assert written.error is None, written.error
        return {}

    graph = StateGraph(FilesystemState, context_schema=RuntimeModelContext)
    graph.add_node("observe", observe)
    graph.add_edge(START, "observe")
    graph.add_edge("observe", END)
    return object(), graph.compile(checkpointer=get_checkpointer())


@pytest.mark.parametrize("client", [True], indirect=True)
def test_shared_files_through_real_stream_checkpoint_and_deletion(client, user_id):
    from juena_core.filesystem import get_filesystem
    from juena_core.filesystem.storage import WorkspaceOwnershipError
    thread_id = _create_chat(client, SIMULATOR)
    base = f"/threads/{thread_id}/filesystem"
    content = b"\xef\xbb\xbfvalue = 42\r\n"
    response = client.post(f"/{SIMULATOR}/stream_with_files", data={
        "thread_id": thread_id, "message": "Inspect this file",
    }, files=[("attachments", ("code.py", content, "text/plain"))])
    assert response.status_code == 200, response.text
    assert '"type": "error"' not in response.text, response.text
    assert client.get(base + "/download", params={"path": "/workspace/nested/observed.py"}).content == content.replace(b"\r\n", b"\n")
    preview = client.get(base + "/file", params={"path": "/workspace/uploads/code.py"}).json()
    assert preview["editable"] is True
    payload = {"path": preview["path"], "revision": preview["revision"], "text": "value = 99\n"}
    assert client.put(base + "/file", json=payload).status_code == 200
    assert client.put(base + "/file", json=payload).status_code == 409
    assert client.get(base + "/file", params={"path": "/findings/committed.md"}).json()["editable"] is False
    fs = get_filesystem()
    files = fs.for_thread(str(user_id), thread_id)
    with files.guard(operation=True, shared=True):
        assert client.delete(f"/threads/{thread_id}").status_code == 409
    fs.active_workspaces = lambda: {files.workspace_id}
    assert client.delete(f"/threads/{thread_id}").status_code == 409
    fs.active_workspaces = set
    assert client.delete(f"/threads/{thread_id}").status_code == 200
    assert client.get(base).status_code == 404
    with pytest.raises(WorkspaceOwnershipError):
        files.ensure()


@pytest.mark.parametrize("client", [True], indirect=True)
def test_checkpointed_interrupt_blocks_edits_after_stream_returns(client, user_id):
    from juena_core.filesystem import get_filesystem
    thread_id = _create_chat(client, SIMULATOR)
    files = get_filesystem().for_thread(str(user_id), thread_id)
    files.ensure()
    path = files.add_upload("code.py", b"original")
    response = client.post(f"/{SIMULATOR}/stream", json={"thread_id": thread_id, "message": "pause"})
    assert response.status_code == 200, response.text
    assert "clarification_required" in response.text
    result = client.put(f"/threads/{thread_id}/filesystem/file", json={
        "path": path, "text": "changed", "revision": files.preview(path, {})["revision"],
    })
    assert result.status_code == 409, result.text
    assert files.store.paused(files.workspace_id)


def test_delta_files_survive_handoff_snapshots_and_restart(monkeypatch, user_id, tmp_path):
    from deepagents.backends.utils import create_file_data
    from deepagents.middleware.filesystem import FilesystemState
    from langchain_core.messages import AIMessage, HumanMessage

    from juena_core.agents.backends import FindingsStateBackend
    from juena_core.agents.delegation import SpecialistDelegate
    from juena_core.filesystem.service import ConversationFilesystem
    from juena_core.filesystem.storage import WorkspaceStore

    monkeypatch.setattr(config_module, "_settings", make_settings())
    monkeypatch.setattr(registry_module, "_agent_registry", {})
    monkeypatch.setattr(registry_module, "_agent_factories", {})
    filesystem = ConversationFilesystem(
        WorkspaceStore(tmp_path / "files"), identity_secret="delta-test-identity-secret-32-characters",
    )

    async def factory(provider, model):
        async def record(state):
            backend = FindingsStateBackend()
            result = await backend.awrite("/note.md", "Specialist evidence")
            assert result.error is None
            return {"messages": [AIMessage(content="Recorded the finding.")]}

        specialist = StateGraph(FilesystemState)
        specialist.add_node("record", record)
        specialist.add_edge(START, "record")
        specialist.add_edge("record", END)
        parent = StateGraph(FilesystemState)
        parent.add_node("delegate", SpecialistDelegate(specialist.compile(), name="specialist"))
        parent.add_edge(START, "delegate")
        parent.add_edge("delegate", END)
        return object(), parent.compile(checkpointer=get_checkpointer())

    registry_module.register_agent_factory(SIMULATOR, factory, set_as_default=True)
    thread_id = None
    expected_input = "pasted input"

    def assert_files(client):
        base = f"/threads/{thread_id}/filesystem"
        listing = client.get(base, params={"path": "/findings"})
        assert listing.status_code == 200, listing.text
        assert [entry["path"] for entry in listing.json()["entries"]] == ["/findings/note.md"]
        for path, text in (("/findings/note.md", "Specialist evidence"),
                           ("/inputs/message.txt", expected_input)):
            preview = client.get(base + "/file", params={"path": path})
            assert preview.status_code == 200, preview.text
            assert preview.json()["text"] == text
            assert preview.json()["editable"] is False
            assert client.get(base + "/download", params={"path": path}).content == text.encode()
            assert client.put(base + "/file", json={
                "path": path, "text": "overwrite", "revision": "a" * 64,
            }).status_code == 403
        # Migration must see legacy uploads in the reconstructed state too.
        assert client.get(base + "/download", params={
            "path": "/workspace/uploads/legacy.txt",
        }).content == b"legacy bytes"
        assert client.get(base + "/download", params={
            "path": "/inputs/uploads/legacy.txt",
        }).content == b"legacy bytes"

    # A second lifespan recreates the connection pool, saver, and compiled agent.
    for restart in (False, True):
        app = create_app(principal=local_principal(user_id=user_id), filesystem=filesystem)
        with TestClient(app) as client:
            if restart:
                assert_files(client)
                continue
            thread_id = _create_chat(client, SIMULATOR)
            config = {"configurable": {"thread_id": thread_id}}

            async def seed():
                graph = await registry_module.get_agent(SIMULATOR)
                await graph.ainvoke({"messages": [HumanMessage(content="Record evidence")], "files": {
                    "/inputs/message.txt": create_file_data(expected_input),
                    "/inputs/uploads/legacy.txt": create_file_data("legacy bytes"),
                }}, config)
                raw = await get_checkpointer().aget(config)
                assert not raw["channel_values"].get("files")

            client.portal.call(seed)
            assert_files(client)

            async def update_inputs():
                graph = await registry_module.get_agent(SIMULATOR)
                # Cross the real FilesystemState's 50-step snapshot interval,
                # then leave a newer delta that raw snapshot reads would miss.
                for number in range(55):
                    await graph.aupdate_state(config, {"files": {
                        "/inputs/message.txt": create_file_data(f"revision {number}"),
                    }})

            client.portal.call(update_inputs)
            expected_input = "revision 54"
            assert_files(client)


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


@pytest.mark.parametrize("client", [True], indirect=True)
def test_deleted_conversation_identity_is_refused_by_create_and_stream(client, user_id):
    thread_id = _create_chat(client, SIMULATOR)
    assert client.delete(f"/threads/{thread_id}").status_code == 200
    assert client.get(f"/chats/{thread_id}").status_code == 410
    response = client.post("/chats", json={"thread_id": thread_id, "agent_id": SIMULATOR, "title": "stale tab"})
    assert response.status_code == 410, response.text
    response = client.post(f"/{SIMULATOR}/stream", json={"thread_id": thread_id, "message": "stale tab"})
    assert response.status_code == 410, response.text
    assert response.json()["detail"]["code"] == "conversation_deleted"
    assert all(chat["thread_id"] != thread_id for chat in client.get("/chats").json())
    assert _create_chat(client, SIMULATOR) != thread_id


@pytest.mark.parametrize("client", [True], indirect=True)
def test_begin_turn_waits_for_short_file_panel_operation(client, user_id):
    from concurrent.futures import ThreadPoolExecutor
    from time import sleep
    from juena_core.filesystem import get_filesystem
    thread_id = _create_chat(client, SIMULATOR)
    files = get_filesystem().for_thread(str(user_id), thread_id)
    files.ensure()
    files.add_upload("code.py", b"print(1)")
    with ThreadPoolExecutor(max_workers=1) as pool:
        with files.guard():
            response = pool.submit(client.post, f"/{SIMULATOR}/stream", json={"thread_id": thread_id, "message": "Inspect"})
            sleep(0.1)
            assert not response.done()
        result = response.result(timeout=10)
    assert result.status_code == 200
    assert '"type": "error"' not in result.text, result.text


@pytest.mark.parametrize("client", [True], indirect=True)
@pytest.mark.parametrize("during_send", [False, True])
def test_cancelled_stream_releases_operation_and_does_not_pin_scratch(client, user_id, during_send):
    import asyncio
    import json
    import os
    from juena_core.filesystem import get_filesystem
    thread_id = _create_chat(client, SIMULATOR)
    files = get_filesystem().for_thread(str(user_id), thread_id)
    files.ensure()
    files.path("/workspace/.scratch/temp.py").write_text("temporary")
    body = json.dumps({"thread_id": thread_id, "message": "cancel this response"}).encode()

    async def disconnect():
        received = False
        first, left = asyncio.Event(), asyncio.Event()
        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            await left.wait()
            return {"type": "http.disconnect"}
        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first.set()
                if during_send:
                    await asyncio.Event().wait()
        scope = {"type": "http", "http_version": "1.1", "method": "POST",
                 "scheme": "http", "path": f"/{SIMULATOR}/stream", "root_path": "",
                 "query_string": b"", "headers": [(b"content-type", b"application/json")],
                 "server": ("localhost", 80), "client": ("127.0.0.1", 1)}
        serving = asyncio.create_task(client.app(scope, receive, send))
        await asyncio.wait_for(first.wait(), 5)
        await asyncio.sleep(0.1)
        left.set()
        await asyncio.wait_for(serving, 5)
    client.portal.call(disconnect)
    assert not files.store.paused(files.workspace_id)
    with files.guard(operation=True):
        pass
    os.utime(files.store.workspace_root(files.workspace_id), (1, 1))
    assert files.store.cleanup_expired(set()) == 1


@pytest.mark.parametrize("client", [True], indirect=True)
def test_create_waits_for_deletion_then_rechecks_tombstone(client, user_id):
    from concurrent.futures import ThreadPoolExecutor
    from time import sleep
    from juena_core.filesystem import get_filesystem
    thread_id = str(uuid4())
    files = get_filesystem().for_thread(str(user_id), thread_id)
    files.ensure()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with files.guard(operation=True):
            create = pool.submit(client.post, "/chats", json={"thread_id": thread_id, "agent_id": SIMULATOR})
            sleep(0.1)
            assert not create.done()
            files.store.delete(files.workspace_id, files.tenant_id)
        assert create.result(timeout=5).status_code == 410
    assert all(chat["thread_id"] != thread_id for chat in client.get("/chats").json())
