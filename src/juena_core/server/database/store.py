"""Stub for 01/CP3. Ported from ``juena/server/database/store.py``
(00-BOUNDARY.md, *Moves whole*)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from langgraph.store.postgres.aio import AsyncPostgresStore

__all__ = ["get_store", "store_lifespan"]


def get_store() -> AsyncPostgresStore:
    raise NotImplementedError("juena_core.server.database.store.get_store lands in 01/CP3")


async def store_lifespan() -> AsyncGenerator[AsyncPostgresStore, None]:
    raise NotImplementedError("juena_core.server.database.store.store_lifespan lands in 01/CP3")
    yield  # pragma: no cover - satisfies the generator signature
