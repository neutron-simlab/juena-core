"""Stub for 01/CP3. Ported from ``juena/server/database/checkpointer.py``
(00-BOUNDARY.md, *Moves whole*)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

__all__ = ["get_checkpointer", "checkpointer_lifespan"]


def get_checkpointer() -> AsyncPostgresSaver:
    raise NotImplementedError("juena_core.server.database.checkpointer.get_checkpointer lands in 01/CP3")


async def checkpointer_lifespan() -> AsyncGenerator[AsyncPostgresSaver, None]:
    raise NotImplementedError("juena_core.server.database.checkpointer.checkpointer_lifespan lands in 01/CP3")
    yield  # pragma: no cover - satisfies the generator signature
