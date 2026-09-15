"""Stub for 01/CP4. Ported from ``juena/server/agent/registry.py`` verbatim,
including ``register_agent_factory(set_as_default=)`` — this is the
convention v2 inherits (00-BOUNDARY.md, *Moves whole*).

``DEFAULT_AGENT`` starts unset: a shared package must not silently choose one
application's agent. The application module registers an explicit default at
import time. Read it through ``get_default_agent()``, never by importing the
name directly, because registration rebinds it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

__all__ = [
    "DEFAULT_AGENT",
    "AgentInstance",
    "register_agent_factory",
    "get_default_agent",
    "get_agent",
    "get_agent_resources",
    "shutdown_agents",
    "list_registered_agents",
]

DEFAULT_AGENT: str | None = None

AgentInstance = Any

_agent_registry: dict[str, tuple[Any, Any]] = {}
_agent_factories: dict[str, Callable[[str, str], Awaitable[tuple[Any, Any]]]] = {}


def register_agent_factory(
    agent_id: str,
    factory: Callable[[str, str], Awaitable[tuple[Any, Any]]],
    set_as_default: bool = False,
) -> None:
    raise NotImplementedError("juena_core.server.agent.registry.register_agent_factory lands in 01/CP4")


def get_default_agent() -> str:
    raise NotImplementedError("juena_core.server.agent.registry.get_default_agent lands in 01/CP4")


def _normalize_provider_model(provider: str | None, model: str | None) -> tuple[str, str]:
    raise NotImplementedError("juena_core.server.agent.registry._normalize_provider_model lands in 01/CP4")


async def get_agent(agent_id: str, provider: str | None = None, model: str | None = None) -> Any:
    raise NotImplementedError("juena_core.server.agent.registry.get_agent lands in 01/CP4")


def get_agent_resources(agent_id: str) -> AgentInstance:
    raise NotImplementedError("juena_core.server.agent.registry.get_agent_resources lands in 01/CP4")


async def shutdown_agents() -> None:
    raise NotImplementedError("juena_core.server.agent.registry.shutdown_agents lands in 01/CP4")


def list_registered_agents() -> list[str]:
    raise NotImplementedError("juena_core.server.agent.registry.list_registered_agents lands in 01/CP4")
