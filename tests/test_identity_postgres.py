"""CP3 persistence and identity, against a real Postgres.

Requires the throwaway database:
``docker compose -f tests/compose.postgres.yml up -d --wait``

These do not skip when Postgres is unreachable. Every property here is about
what the database actually does — which columns exist, which constraint
fires, whether a foreign key resolves — and a skip would turn "not measured"
into "passed".
"""

from __future__ import annotations

import inspect as inspect_module
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import inspect, text

from juena_core.config import CoreSettings
from juena_core.schema.llm_models import BlabladorModelName
from juena_core.server import identity as identity_module
from juena_core.server.chat.repository import (
    ChatNotFoundError,
    chat_to_dict,
    ensure_owned_chat,
    get_owned_chat,
    list_owned_chats,
)
from juena_core.server.database import connection as connection_module
from juena_core.server.database.models import Base, Chat, User
from juena_core.server.identity import (
    Principal,
    create_auth_session,
    ensure_principal_row,
    fixed_principal,
    hash_session_token,
    local_principal,
    refuse_published_api,
    revoke_auth_session,
    revoke_sessions_for_subject,
    session_principal,
    upsert_principal,
)

DSN = "postgresql://juena_core_test:juena_core_test@127.0.0.1:55432/juena_core_test"


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
        "ARTIFACT_ROOT": Path("/tmp/juena-core-test-artifacts"),
        "AUDIT_FILE": Path("/tmp/juena-core-test-artifacts/audit.jsonl"),
        "LOG_LEVEL": "INFO",
        "LOG_DIR": Path("/tmp/juena-core-test-logs"),
        "BIND_HOST": "127.0.0.1",
        "API_PUBLISHED": False,
    }
    values.update(overrides)
    return CoreSettings(**values)  # type: ignore[arg-type]


@asynccontextmanager
async def _running_database(config: CoreSettings):
    """``database_lifespan`` with core settings installed for its duration."""

    original = connection_module.settings
    connection_module.settings = lambda: config  # type: ignore[assignment]
    try:
        async with connection_module.database_lifespan() as engine:
            yield engine
    finally:
        connection_module.settings = original  # type: ignore[assignment]


@pytest_asyncio.fixture
async def database(monkeypatch):
    """A live schema, dropped and rebuilt so each test starts from nothing."""

    config = make_settings()
    monkeypatch.setattr(identity_module, "settings", lambda: config)
    async with _running_database(config) as engine:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
            await connection.run_sync(Base.metadata.create_all)
        yield engine


@pytest_asyncio.fixture
async def db_session(database):
    factory = connection_module.get_session_factory()
    async with factory() as session:
        yield session


async def _principal(session, *, subject: str = "subject-1") -> Principal:
    principal = await upsert_principal(
        session,
        subject=subject,
        issuer="https://idp.example/entity",
        email="scientist@example.org",
        display_name="A Scientist",
    )
    await session.commit()
    return principal


@pytest.mark.asyncio
async def test_renamed_attributes_keep_their_original_columns(database) -> None:
    """``create_all`` never renames a column, so the attributes moved instead."""

    async with database.connect() as connection:
        columns = await connection.run_sync(
            lambda sync: {column["name"] for column in inspect(sync).get_columns("users")}
        )

    assert {"saml_subject", "idp_entity_id"} <= columns
    assert not {"subject", "issuer"} & columns


@pytest.mark.asyncio
async def test_chats_record_their_agent_and_refuse_null(database) -> None:
    async with database.connect() as connection:
        agent_column = await connection.run_sync(
            lambda sync: next(
                column
                for column in inspect(sync).get_columns("chats")
                if column["name"] == "agent_id"
            )
        )

    assert agent_column["nullable"] is False


@pytest.mark.asyncio
async def test_session_round_trip_and_revocation(db_session) -> None:
    principal = await _principal(db_session)
    token = await create_auth_session(db_session, principal)
    await db_session.commit()

    resolved = await session_principal(session_token=token, session=db_session)
    assert resolved == principal
    assert resolved.issuer == "https://idp.example/entity"

    stored = await db_session.execute(
        text("SELECT token_hash FROM auth_sessions WHERE user_id = :user_id"),
        {"user_id": principal.id},
    )
    assert stored.scalar_one() == hash_session_token(token)

    await revoke_auth_session(db_session, token)
    await db_session.commit()
    with pytest.raises(Exception) as excinfo:
        await session_principal(session_token=token, session=db_session)
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_revoking_a_subject_closes_every_live_session(db_session) -> None:
    principal = await _principal(db_session)
    first = await create_auth_session(db_session, principal)
    second = await create_auth_session(db_session, principal)
    await db_session.commit()

    assert await revoke_sessions_for_subject(db_session, principal.subject) == 2
    await db_session.commit()

    for token in (first, second):
        with pytest.raises(Exception) as excinfo:
            await session_principal(session_token=token, session=db_session)
        assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_a_chat_belongs_to_one_agent(db_session) -> None:
    principal = await _principal(db_session)

    chat = await ensure_owned_chat(db_session, principal.id, None, "simulator")
    await db_session.commit()
    assert chat.agent_id == "simulator"
    assert chat_to_dict(chat)["agent_id"] == "simulator"

    # Same thread, same owner, other graph: refused as missing, not as denied.
    with pytest.raises(ChatNotFoundError):
        await ensure_owned_chat(db_session, principal.id, chat.thread_id, "advanced_mode")
    with pytest.raises(ChatNotFoundError):
        await get_owned_chat(db_session, principal.id, chat.thread_id, agent_id="advanced_mode")

    assert await get_owned_chat(db_session, principal.id, chat.thread_id, agent_id="simulator")


@pytest.mark.asyncio
async def test_another_users_thread_is_missing_rather_than_forbidden(db_session) -> None:
    owner = await _principal(db_session, subject="owner")
    intruder = await _principal(db_session, subject="intruder")
    chat = await ensure_owned_chat(db_session, owner.id, None, "simulator")
    await db_session.commit()

    with pytest.raises(ChatNotFoundError):
        await ensure_owned_chat(db_session, intruder.id, chat.thread_id, "simulator")


@pytest.mark.asyncio
async def test_listing_separates_the_agents(db_session) -> None:
    principal = await _principal(db_session)
    await ensure_owned_chat(db_session, principal.id, "thread-sim", "simulator")
    await ensure_owned_chat(db_session, principal.id, "thread-batch", "advanced_mode")
    await db_session.commit()

    everything = await list_owned_chats(db_session, principal.id)
    simulator_only = await list_owned_chats(db_session, principal.id, agent_id="simulator")

    assert {chat.thread_id for chat in everything} == {"thread-sim", "thread-batch"}
    assert [chat.thread_id for chat in simulator_only] == ["thread-sim"]


@pytest.mark.asyncio
async def test_local_principal_can_own_a_conversation_after_row_upsert(database) -> None:
    """CP3 proves the primitive; CP4 wires it into application startup."""

    user_id = uuid4()
    dependency = local_principal(user_id=user_id, display_name="Local User")
    principal = fixed_principal(dependency)
    assert principal is not None
    assert await dependency() is principal

    await ensure_principal_row(principal)

    factory = connection_module.get_session_factory()
    async with factory() as session:
        assert await session.get(User, user_id) is not None
        session.add(Chat(thread_id="local-thread", user_id=user_id, agent_id="simulator"))
        await session.commit()


@pytest.mark.asyncio
async def test_principal_row_upsert_is_idempotent_and_catches_a_changed_id(database) -> None:
    principal = Principal(
        id=uuid4(), subject="local", issuer="local", email=None, display_name=None
    )
    await ensure_principal_row(principal)
    await ensure_principal_row(principal)

    moved = Principal(
        id=uuid4(), subject="local", issuer="local", email=None, display_name=None
    )
    with pytest.raises(RuntimeError, match="does not match the stored id"):
        await ensure_principal_row(moved)


def test_a_fixed_principal_refuses_a_published_api(monkeypatch) -> None:
    monkeypatch.setattr(identity_module, "settings", lambda: make_settings(API_PUBLISHED=True))
    with pytest.raises(RuntimeError, match="API_PUBLISHED"):
        local_principal(user_id=uuid4())


def test_a_fixed_principal_refuses_a_non_loopback_bind(monkeypatch) -> None:
    monkeypatch.setattr(identity_module, "settings", lambda: make_settings(BIND_HOST="0.0.0.0"))
    with pytest.raises(RuntimeError, match="BIND_HOST"):
        local_principal(user_id=uuid4())


def test_a_real_provider_is_not_subject_to_the_fixed_principal_guard(monkeypatch) -> None:
    """Publishing the API is exactly what a real identity provider is for."""

    monkeypatch.setattr(identity_module, "settings", lambda: make_settings(API_PUBLISHED=True))
    assert fixed_principal(session_principal) is None
    refuse_published_api(session_principal)


def test_reading_a_chat_cannot_skip_the_agent_check_by_omission() -> None:
    """A default here would let a resume route drop the check silently."""

    parameter = inspect_module.signature(get_owned_chat).parameters["agent_id"]
    assert parameter.default is inspect_module.Parameter.empty
    assert parameter.kind is inspect_module.Parameter.KEYWORD_ONLY


def test_the_session_cookie_name_is_unchanged() -> None:
    assert identity_module.SESSION_COOKIE_NAME == "juena_session"


def test_principal_ids_are_uuids() -> None:
    principal = Principal(
        id=uuid4(), subject="s", issuer="i", email=None, display_name=None
    )
    assert isinstance(principal.id, UUID)
