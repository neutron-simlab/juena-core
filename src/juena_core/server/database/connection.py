"""Stub for 01/CP3. Ported from ``juena/server/database/connection.py``
(00-BOUNDARY.md, *Moves whole*)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

__all__ = [
    "psycopg_database_url",
    "sqlalchemy_database_url",
    "get_pool",
    "get_session_factory",
    "get_db_session",
    "database_lifespan",
]

_DSN_PREFIXES = ("postgresql+psycopg://", "postgresql://", "postgres://")


def _strip_dsn_scheme(database_url: str) -> str:
    raise NotImplementedError("juena_core.server.database.connection._strip_dsn_scheme lands in 01/CP3")


def psycopg_database_url(database_url: str) -> str:
    raise NotImplementedError("juena_core.server.database.connection.psycopg_database_url lands in 01/CP3")


def sqlalchemy_database_url(database_url: str) -> str:
    raise NotImplementedError("juena_core.server.database.connection.sqlalchemy_database_url lands in 01/CP3")


def get_pool() -> AsyncConnectionPool:
    raise NotImplementedError("juena_core.server.database.connection.get_pool lands in 01/CP3")


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    raise NotImplementedError("juena_core.server.database.connection.get_session_factory lands in 01/CP3")


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    raise NotImplementedError("juena_core.server.database.connection.get_db_session lands in 01/CP3")
    yield  # pragma: no cover - satisfies the generator signature


async def database_lifespan() -> AsyncGenerator[AsyncEngine, None]:
    raise NotImplementedError("juena_core.server.database.connection.database_lifespan lands in 01/CP3")
    yield  # pragma: no cover - satisfies the generator signature
