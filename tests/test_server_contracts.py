"""CP4 contracts that hold without a database.

The application surface core promises: which routes exist, which tables it
owns, what it refuses to serve, and how the pieces behind the routes behave.
Anything that needs a live Postgres is in ``test_server_routes_postgres.py``.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command, Interrupt
from starlette.datastructures import Headers, UploadFile

from juena_core import config as config_module
from juena_core.artifacts import ARTIFACT_MESSAGE_KEY, ArtifactStore, set_artifact_store_for_tests
from juena_core.config import CoreSettings
from juena_core.schema.interrupts import CLARIFICATION_KIND, ClarificationResumeInput
from juena_core.schema.llm_models import BlabladorModelName
from juena_core.server import interrupts as interrupts_module
from juena_core.server.agent import registry as registry_module
from juena_core.server.agent.input_handler import AgentInputHandler
from juena_core.server.agent.registry import get_agent, get_default_agent, list_registered_agents
from juena_core.server.api.endpoints import DEFAULT_CLOSING_NOTE
from juena_core.server.chat.input_constants import DISPLAY_TEXT_KEY
from juena_core.server.chat.inputs import prepare_code_chat_turn_inputs
from juena_core.server.chat.input_utils import build_inputs_manifest
from juena_core.server.database.models import Base
from juena_core.server.errors import AgentNotFoundError
from juena_core.server.identity import local_principal, session_principal
from juena_core.server.interrupts import (
    ResumeError,
    build_resume_command,
    classify_interrupt,
    interrupt_event,
    register_interrupt_kind,
    registered_interrupt_kinds,
)
from juena_core.server.service import create_app
from juena_core.server.streaming.processor import StreamEventProcessor, StreamPolicy
from juena_core.server.utils import langchain_to_chat_message

#: Every path ``create_app`` must serve. ``pending-approval`` is the older
#: spelling of ``pending-interrupt``, kept so a client built before the
#: clarification kind keeps working.
EXPECTED_ROUTES = {
    "/chats",
    "/chats/{thread_id}",
    "/stream",
    "/stream_with_files",
    "/resume",
    "/{agent_id}/stream",
    "/{agent_id}/stream_with_files",
    "/{agent_id}/resume",
    "/threads/{thread_id}",
    "/threads/{thread_id}/pending-interrupt",
    "/threads/{thread_id}/pending-approval",
    "/artifacts/{artifact_id}",
    "/health",
}


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
        "DATABASE_URL": None,
        "DATABASE_POOL_MAX_SIZE": 5,
        "SESSION_TTL_HOURS": 8,
        "SESSION_COOKIE_SECURE": False,
        "FALLBACK_PROVIDER": None,
        "EXECUTE_TIMEOUT_SECONDS": 600,
        "ARTIFACT_ROOT": Path("/tmp/juena-core-contract-artifacts"),
        "AUDIT_FILE": Path("/tmp/juena-core-contract-artifacts/audit.jsonl"),
        "LOG_LEVEL": "INFO",
        "LOG_DIR": Path("/tmp/juena-core-contract-logs"),
        "BIND_HOST": "127.0.0.1",
        "API_PUBLISHED": False,
    }
    values.update(overrides)
    return CoreSettings(**values)  # type: ignore[arg-type]


@pytest.fixture
def configured(monkeypatch):
    """Install core settings process-wide for one test.

    Patching the module global rather than each importer's reference: every
    module holds the same ``settings`` function, and that function reads this.
    """

    config = make_settings()
    monkeypatch.setattr(config_module, "_settings", config)
    return config


@pytest.fixture
def local(configured):
    return local_principal(user_id=uuid4())


def route_paths(app: Any) -> set[str]:
    """Every path the application serves.

    Read from the OpenAPI schema rather than by walking ``app.routes``: since
    FastAPI 0.141 an included router is one opaque ``_IncludedRouter`` entry
    there, so the plan's ``sorted(r.path for r in app.routes)`` sees only
    ``/health``. The schema is the public surface and flattens inclusion.
    """

    return set(app.openapi()["paths"])


# --------------------------------------------------------------------------
# The application surface
# --------------------------------------------------------------------------


def test_create_app_serves_exactly_the_documented_routes(local) -> None:
    assert route_paths(create_app(principal=local)) == EXPECTED_ROUTES


def test_nothing_under_auth_is_served(local) -> None:
    """SAML is not here. An application that has one adds its own router."""

    assert not [path for path in route_paths(create_app(principal=local)) if "/auth" in path]


def test_an_application_router_is_included(local) -> None:
    from fastapi import APIRouter

    extra = APIRouter(prefix="/auth", tags=["auth"])

    @extra.get("/me")
    async def me() -> dict:
        return {}

    app = create_app(principal=local, extra_routers=(extra,))
    assert "/auth/me" in route_paths(app)


def _resume_body_schema(app: Any) -> dict:
    return app.openapi()["paths"]["/resume"]["post"]["requestBody"]["content"]["application/json"][
        "schema"
    ]


def test_the_resume_body_defaults_to_the_clarification_arm_alone(local) -> None:
    """An application with no other interrupt kind gets a union of one.

    Offering an approval arm it can never raise would describe a pause that
    cannot happen — and would accept a body no registered kind can resume.
    """

    schema = _resume_body_schema(create_app(principal=local))
    assert schema["$ref"].endswith("/ClarificationResumeInput")


def test_an_application_supplies_its_own_resume_union(local) -> None:
    """The union is a parameter because its arms are the application's: core
    cannot name an approval whose model lives in juena-chatbot."""

    from typing import Annotated, Literal

    from pydantic import Field

    from juena_core.schema.interrupts import ResumeBase

    class ApprovalResumeInput(ResumeBase):
        kind: Literal["execute_approval"] = "execute_approval"
        decision: Literal["approve", "edit", "reject"]

    union = Annotated[
        ApprovalResumeInput | ClarificationResumeInput, Field(discriminator="kind")
    ]
    schema = _resume_body_schema(create_app(principal=local, resume_input=union))
    assert set(schema["discriminator"]["mapping"]) == {"execute_approval", "clarification"}


def test_extra_lifespans_unwind_before_the_agents_they_may_be_using(configured, monkeypatch) -> None:
    """An application's own shutdown runs first; `shutdown_agents` runs last.

    Found during the plan-02 cutover. juena-chatbot ran, in this order,
    `research_runner.shutdown()` — which waits for specialists running detached
    from any request — and only then closed its agents. Entering the extra
    lifespans inside the `try` whose `finally` closed the agents inverted that:
    the clients would be pulled out from under work still finishing.

    The database lifespan is still open around both, so the original constraint
    — close agents before the pools they may be using — continues to hold.
    """

    import contextlib

    from juena_core.server import service as service_module

    events: list[str] = []

    @contextlib.asynccontextmanager
    async def noop():
        yield

    @contextlib.asynccontextmanager
    async def application_work():
        try:
            yield
        finally:
            events.append("extra lifespan exited")

    async def record_shutdown():
        events.append("agents closed")

    for name in ("database_lifespan", "checkpointer_lifespan", "store_lifespan"):
        monkeypatch.setattr(service_module, name, noop)
    monkeypatch.setattr(service_module, "shutdown_agents", record_shutdown)

    # A real identity provider, so no `users` row is upserted at startup:
    # this test is about ordering, and the database lifespans are stubbed.
    app = create_app(principal=session_principal, extra_lifespans=(application_work,))

    async def run() -> None:
        async with app.router.lifespan_context(app):
            events.append("serving")

    asyncio.run(run())

    assert events == ["serving", "extra lifespan exited", "agents closed"]


def test_core_owns_exactly_three_tables() -> None:
    """A model never imported is never created, and fails at first write.

    Pinning the set is what turns that into a failing test instead of a
    missing table discovered by a user.
    """

    assert set(Base.metadata.tables) == {"users", "auth_sessions", "chats"}


def test_importing_the_service_registers_no_agent() -> None:
    """Registering an agent requires importing *its* module, not this one."""

    assert list_registered_agents() == []


def test_serving_without_a_registered_default_fails_loudly(configured) -> None:
    """And says what to do, rather than reporting a missing agent."""

    with pytest.raises(RuntimeError, match="must import its agent module") as excinfo:
        get_default_agent()
    assert not isinstance(excinfo.value, AgentNotFoundError)


def test_registering_a_default_rebinds_it(configured, monkeypatch) -> None:
    monkeypatch.setattr(registry_module, "_agent_factories", {})
    monkeypatch.setattr(registry_module, "DEFAULT_AGENT", None)

    async def factory(provider: str, model: str):  # pragma: no cover - never called
        raise AssertionError

    registry_module.register_agent_factory("simulator", factory, set_as_default=True)
    assert get_default_agent() == "simulator"


@pytest.mark.asyncio
async def test_concurrent_first_requests_build_one_agent(configured, monkeypatch) -> None:
    """The process cache must not leak one of two simultaneous builds."""

    monkeypatch.setattr(registry_module, "_agent_registry", {})
    monkeypatch.setattr(registry_module, "_agent_factories", {})
    monkeypatch.setattr(registry_module, "_agent_creation_locks", {})
    monkeypatch.setattr(registry_module, "DEFAULT_AGENT", None)

    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    graph = object()

    async def factory(provider: str, model: str):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return object(), graph

    registry_module.register_agent_factory("simulator", factory, set_as_default=True)
    first = asyncio.create_task(get_agent("simulator"))
    await started.wait()
    second = asyncio.create_task(get_agent("simulator"))
    await asyncio.sleep(0)
    release.set()

    assert await asyncio.gather(first, second) == [graph, graph]
    assert calls == 1


# --------------------------------------------------------------------------
# The publication guard
# --------------------------------------------------------------------------


def test_create_app_refuses_a_fixed_principal_on_a_published_api(monkeypatch) -> None:
    """The second gate: an application may build its principal before it
    decides how to serve it, and this is the moment the port is about to open."""

    monkeypatch.setattr(config_module, "_settings", make_settings())
    principal = local_principal(user_id=uuid4())

    monkeypatch.setattr(config_module, "_settings", make_settings(API_PUBLISHED=True))
    with pytest.raises(RuntimeError, match="API_PUBLISHED"):
        create_app(principal=principal)


def test_create_app_refuses_a_fixed_principal_on_a_non_loopback_bind(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "_settings", make_settings())
    principal = local_principal(user_id=uuid4())

    monkeypatch.setattr(config_module, "_settings", make_settings(BIND_HOST="0.0.0.0"))
    with pytest.raises(RuntimeError, match="BIND_HOST"):
        create_app(principal=principal)


def test_a_real_identity_provider_stays_publishable(monkeypatch) -> None:
    """Publishing the API is exactly what a real identity provider is for."""

    monkeypatch.setattr(
        config_module, "_settings", make_settings(API_PUBLISHED=True, BIND_HOST="0.0.0.0")
    )
    assert route_paths(create_app(principal=session_principal)) == EXPECTED_ROUTES


# --------------------------------------------------------------------------
# The interrupt-kind registry
# --------------------------------------------------------------------------


def _clarification(question: str = "Which sample?", interrupt_id: str = "i-1") -> Interrupt:
    return Interrupt(value={"kind": CLARIFICATION_KIND, "question": question}, id=interrupt_id)


def test_core_registers_only_the_clarification_kind() -> None:
    assert registered_interrupt_kinds() == [CLARIFICATION_KIND]


def test_a_clarification_is_classified_and_rendered() -> None:
    interrupt = _clarification()
    assert classify_interrupt(interrupt) == CLARIFICATION_KIND
    payload = interrupt_event(interrupt)
    assert payload == {
        "type": "clarification_required",
        "interrupt_id": "i-1",
        "asked_by": "",
        "question": "Which sample?",
        "options": [],
    }


def test_an_unregistered_interrupt_is_neither_classified_nor_rendered() -> None:
    """A HumanInTheLoop approval reaches core unrecognised until an
    application registers it — which is what keeps the sandbox out of here."""

    approval = Interrupt(value={"action_requests": [{"name": "execute"}]}, id="i-2")
    assert classify_interrupt(approval) is None
    assert interrupt_event(approval) is None


def test_a_malformed_clarification_is_not_rendered() -> None:
    assert interrupt_event(Interrupt(value={"kind": CLARIFICATION_KIND}, id="i-3")) is None
    assert interrupt_event(Interrupt(value="not a dict", id="i-4")) is None


def test_an_application_kind_is_rendered_and_resumed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(interrupts_module, "_interrupt_kinds", {})

    def event(interrupt: Interrupt) -> dict | None:
        if not isinstance(interrupt.value, dict) or "action_requests" not in interrupt.value:
            return None
        return {"type": "approval_required", "interrupt_id": interrupt.id}

    def resume(*, interrupt: Interrupt, user_id: str, payload: Any) -> Command:
        return Command(resume={payload.interrupt_id: {"decisions": [payload.decision]}})

    register_interrupt_kind("execute_approval", event=event, resume=resume)

    approval = Interrupt(value={"action_requests": [{"name": "execute"}]}, id="i-5")
    assert classify_interrupt(approval) == "execute_approval"
    assert interrupt_event(approval)["type"] == "approval_required"


@pytest.mark.asyncio
async def test_resume_refuses_an_unregistered_kind(configured) -> None:
    class Payload:
        kind = "nonexistent"
        interrupt_id = "i-6"
        thread_id = "t-1"

    with pytest.raises(ResumeError, match="No interrupt kind is registered"):
        await build_resume_command(agent=None, config={}, user_id="u", payload=Payload())


@pytest.mark.asyncio
async def test_resume_refuses_a_reply_for_a_different_kind(configured, monkeypatch) -> None:
    """A clarification answer must not resume an approval, or vice versa."""

    class _Task:
        interrupts = (Interrupt(value={"action_requests": []}, id="i-7"),)

    class _Snapshot:
        tasks = (_Task(),)

    class _Agent:
        async def aget_state(self, config):
            return _Snapshot()

    payload = ClarificationResumeInput(
        kind="clarification", thread_id="t-1", interrupt_id="i-7", answer="yes"
    )
    with pytest.raises(ResumeError, match="not a pending clarification"):
        await build_resume_command(agent=_Agent(), config={}, user_id="u", payload=payload)


@pytest.mark.asyncio
async def test_a_stale_interrupt_id_is_refused(configured) -> None:
    class _Snapshot:
        tasks = ()

    class _Agent:
        async def aget_state(self, config):
            return _Snapshot()

    payload = ClarificationResumeInput(
        kind="clarification", thread_id="t-1", interrupt_id="gone", answer="yes"
    )
    with pytest.raises(ResumeError, match="stale"):
        await build_resume_command(agent=_Agent(), config={}, user_id="u", payload=payload)


@pytest.mark.asyncio
async def test_a_clarification_resumes_with_the_users_own_words(configured, tmp_path) -> None:
    set_artifact_store_for_tests(ArtifactStore(tmp_path / "a", tmp_path / "audit.jsonl"))
    try:

        class _Task:
            interrupts = (_clarification(interrupt_id="i-8"),)

        class _Snapshot:
            tasks = (_Task(),)

        class _Agent:
            async def aget_state(self, config):
                return _Snapshot()

        payload = ClarificationResumeInput(
            kind="clarification", thread_id="t-1", interrupt_id="i-8", answer="  the cold one  "
        )
        command = await build_resume_command(
            agent=_Agent(), config={}, user_id="u", payload=payload
        )
        assert command.resume == {"i-8": "the cold one"}
    finally:
        set_artifact_store_for_tests(None)


# --------------------------------------------------------------------------
# The stream policy
# --------------------------------------------------------------------------


async def _collect(processor: StreamEventProcessor, event: Any) -> list[dict]:
    return [payload async for payload in processor.process_event(event)]


def _processor(**policy: Any) -> StreamEventProcessor:
    return StreamEventProcessor(
        agent=None, config={}, run_id="r-1", user_input_message="", policy=StreamPolicy(**policy)
    )


@pytest.mark.asyncio
async def test_a_custom_payload_is_forwarded_only_when_declared() -> None:
    """The custom channel carries whatever any middleware wrote, so forwarding
    by default would put internal payloads on a user's wire."""

    event = ("custom", {"type": "sandbox_status", "status": "running"})

    assert await _collect(_processor(), event) == []
    forwarded = await _collect(
        _processor(custom_event_types=frozenset({"sandbox_status"})), event
    )
    assert forwarded == [{"type": "sandbox_status", "status": "running"}]


@pytest.mark.asyncio
async def test_an_undeclared_custom_type_is_dropped_even_alongside_a_declared_one() -> None:
    processor = _processor(custom_event_types=frozenset({"sandbox_status"}))
    assert await _collect(processor, ("custom", {"type": "debug_dump", "rows": 10_000})) == []


@pytest.mark.asyncio
async def test_a_tool_result_becomes_a_status_event() -> None:
    message = ToolMessage(content="42 rows", tool_call_id="c-1", name="search")
    payloads = await _collect(_processor(), ("updates", {"tools": {"messages": [message]}}))
    assert payloads == [
        {
            "type": "status",
            "phase": "tool",
            "label": "Ran search",
            "tool": "search",
            "summary": "42 rows",
        }
    ]


@pytest.mark.asyncio
async def test_a_silent_tool_produces_nothing_at_all() -> None:
    """Not even the one-line preview: its output may be huge or binary-like."""

    message = ToolMessage(content="\x00" * 5000, tool_call_id="c-2", name="execute")
    processor = _processor(silent_tools=frozenset({"execute"}))
    assert await _collect(processor, ("updates", {"tools": {"messages": [message]}})) == []


# --------------------------------------------------------------------------
# Message conversion
# --------------------------------------------------------------------------


def test_a_staged_human_turn_renders_what_the_user_typed() -> None:
    """The model reads the manifest; the transcript shows the typed text."""

    message = HumanMessage(
        content="Inspect `/inputs` before answering.",
        additional_kwargs={DISPLAY_TEXT_KEY: "why is my beam divergent?"},
    )
    assert langchain_to_chat_message(message).content == "why is my beam divergent?"


def test_artifacts_recorded_on_an_ai_message_survive_conversion() -> None:
    message = AIMessage(
        content="Here is the plot.",
        additional_kwargs={ARTIFACT_MESSAGE_KEY: [{"artifact_id": "a-1", "kind": "image"}]},
    )
    converted = langchain_to_chat_message(message)
    assert converted.custom_data["artifacts"] == [{"artifact_id": "a-1", "kind": "image"}]


def test_an_unsupported_message_type_is_refused() -> None:
    with pytest.raises(ValueError, match="Unsupported message type"):
        langchain_to_chat_message(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Run context and manifests
# --------------------------------------------------------------------------


def test_a_run_requires_an_authenticated_user(configured) -> None:
    with pytest.raises(ValueError, match="trusted authenticated user_id is required"):
        AgentInputHandler.build_run_context(thread_id="t-1", user_id=None)


def test_a_run_context_carries_identity_into_the_graph(configured) -> None:
    run = AgentInputHandler.build_run_context(thread_id="t-1", user_id="u-1")
    assert run.config["configurable"]["thread_id"] == "t-1"
    assert run.context.user_id == "u-1"
    assert run.context.thread_id == "t-1"


def test_the_manifest_closing_note_is_the_applications() -> None:
    """Core's default states a fact. What an agent should *do* with these
    paths depends on what that agent can reach, so it is passed in."""

    manifest = build_inputs_manifest(
        "help", (), None, has_thread_uploads=True, closing_note="Ask the simulator first."
    )
    assert manifest.endswith("Ask the simulator first.")
    assert DEFAULT_CLOSING_NOTE not in manifest


@pytest.mark.asyncio
async def test_workspace_view_matches_the_files_sent_to_the_graph() -> None:
    """A process-backed workspace sees manifests and turn helpers too."""

    class Agent:
        async def aget_state(self, config):
            return SimpleNamespace(values={"files": {}})

    upload = UploadFile(
        file=io.BytesIO(b"print('uploaded')\n"),
        filename="uploaded.py",
        headers=Headers({"content-type": "text/x-python"}),
    )
    prepared = await prepare_code_chat_turn_inputs(
        Agent(),  # type: ignore[arg-type]
        config={},  # type: ignore[arg-type]
        message="Please inspect this.\n```python\nprint('pasted')\n```",
        attachments=[upload],
        closing_note="Inspect every staged file.",
    )

    assert prepared is not None
    assert prepared.workspace_files == {
        path: file_data
        for path, file_data in prepared.files_update.items()
        if file_data is not None
    }
    assert set(prepared.workspace_files) == {
        "/inputs/current_code.py",
        "/inputs/current_message.txt",
        "/inputs/uploads/uploaded.py",
        "/inputs/uploads_manifest.md",
    }
