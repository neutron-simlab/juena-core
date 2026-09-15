"""Stub for 01/CP4. Ported from ``juena/server/agent/runtime_model_middleware.py``
(00-BOUNDARY.md, *Moves whole*). ``RuntimeModelContext`` carries ``provider``,
``model``, ``thread_id``, ``user_id`` — half the identity seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware

__all__ = ["RuntimeModelContext", "RuntimeModelMiddleware"]


@dataclass
class RuntimeModelContext:
    """Stub — implemented in 01/CP4."""


def _provider_model_from_context(context: Any) -> tuple[str | None, str | None]:
    raise NotImplementedError("juena_core.server.agent.runtime_model_middleware._provider_model_from_context lands in 01/CP4")


def _get_provider_model_from_request(request: Any) -> tuple[str | None, str | None]:
    raise NotImplementedError("juena_core.server.agent.runtime_model_middleware._get_provider_model_from_request lands in 01/CP4")


class RuntimeModelMiddleware(AgentMiddleware):
    """Stub — implemented in 01/CP4."""
