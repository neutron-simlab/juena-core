"""The registry of compiled agent graphs, keyed by agent id.

An application registers a factory per agent at import time; the first request
for that id runs the factory once and the compiled graph is cached for the
process. Provider and model are handled per invocation through
:class:`~juena_core.server.agent.runtime_model_middleware.RuntimeModelContext`,
so the pair passed to a factory only supplies the defaults used while the graph
is built.

Registering an agent requires importing its module only for that side effect,
in the application's ``service.py``; see ``juena_core.server.service``.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from juena_core.config import settings
from juena_core.llms_providers import get_default_model
from juena_core.log import get_logger
from juena_core.server.errors import AgentNotFoundError

logger = get_logger(__name__)

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

#: The agent id used when a request names none.
#:
#: It starts unset because a shared package must not choose an application's
#: agent for it. ``register_agent_factory(set_as_default=True)`` rebinds it at
#: import time, so read it through :func:`get_default_agent`: a
#: ``from ... import DEFAULT_AGENT`` keeps the value current at that import.
DEFAULT_AGENT: str | None = None

AgentInstance = Any

_agent_registry: dict[str, tuple[AgentInstance, CompiledStateGraph]] = {}
_agent_factories: dict[
    str, Callable[[str, str], Awaitable[tuple[AgentInstance, CompiledStateGraph]]]
] = {}
# FastAPI may receive several requests before the first factory call finishes.
# One lock per id makes the cache's "build once" contract true without making
# unrelated agents wait for each other.
_agent_creation_locks: dict[str, asyncio.Lock] = {}


def register_agent_factory(
    agent_id: str,
    factory: Callable[[str, str], Awaitable[tuple[AgentInstance, CompiledStateGraph]]],
    set_as_default: bool = False,
) -> None:
    """Register a factory that builds one agent's graph.

    Args:
        agent_id: Agent identifier, e.g. ``"simulator"``.
        factory: Async callable taking ``(provider, model)`` and returning
            ``(instance, compiled_graph)``. The instance is whatever the
            application needs to reach past the graph; core stores it and
            hands it back from :func:`get_agent_resources`.
        set_as_default: Serve this agent when a request names none.
    """

    _agent_factories[agent_id] = factory
    logger.info("Registered factory for agent: %s", agent_id)

    if set_as_default:
        global DEFAULT_AGENT
        DEFAULT_AGENT = agent_id
        logger.info("Set %s as the default agent", agent_id)


def get_default_agent() -> str:
    """Return the agent id to use when a request does not name one.

    Returns:
        The id registered with ``set_as_default=True``.

    Raises:
        RuntimeError: No default is registered. Not ``AgentNotFoundError``,
            because the fault is not in the request: the application's service
            module never imported its agent module, and a 404 would send
            whoever is debugging to the request instead of the import.
    """

    if DEFAULT_AGENT is None:
        raise RuntimeError(
            "No default agent is registered. The application's service module must "
            "import its agent module for the registration side effect, and that module "
            "must call register_agent_factory(..., set_as_default=True). "
            f"Registered agents: {list(_agent_factories) or 'none'}."
        )
    return DEFAULT_AGENT


def _normalize_provider_model(provider: str | None, model: str | None) -> tuple[str, str]:
    """Fill in the configured defaults for a missing provider or model."""

    config = settings()
    if provider is None:
        provider = config.DEFAULT_PROVIDER

    if model is None:
        model = get_default_model(provider)
        if not model:
            # An unrecognized provider name resets both halves of the pair.
            provider = config.DEFAULT_PROVIDER
            model = config.DEFAULT_MODEL

    return provider, model


async def get_agent(
    agent_id: str,
    provider: str | None = None,
    model: str | None = None,
) -> CompiledStateGraph:
    """Return an agent's compiled graph, building it on first use.

    Args:
        agent_id: The registered agent id.
        provider: Default provider for the first build only; afterwards model
            selection travels per invocation through runtime context.
        model: Default model for the first build only.

    Returns:
        The agent's compiled graph.

    Raises:
        AgentNotFoundError: The id is not registered, or its factory failed.
    """

    provider, model = _normalize_provider_model(provider, model)

    if agent_id not in _agent_registry:
        if agent_id not in _agent_factories:
            logger.error("Unknown agent requested: %s", agent_id)
            raise AgentNotFoundError(
                agent_id, details={"available_agents": list(_agent_factories)}
            )

        lock = _agent_creation_locks.setdefault(agent_id, asyncio.Lock())
        async with lock:
            # A request waiting for the lock uses the graph the first request
            # just cached instead of constructing and overwriting another one.
            if agent_id not in _agent_registry:
                try:
                    logger.info(
                        "Creating agent %s with default provider=%s model=%s",
                        agent_id,
                        provider,
                        model,
                    )
                    agent_instance, compiled_graph = await _agent_factories[agent_id](
                        provider, model
                    )
                    _agent_registry[agent_id] = (agent_instance, compiled_graph)
                    logger.info("Agent %s created and registered", agent_id)
                except Exception as exc:
                    logger.error("Failed to create agent %s: %s", agent_id, exc)
                    raise AgentNotFoundError(
                        agent_id,
                        details={
                            "error": str(exc),
                            "agent_type": agent_id,
                            "provider": provider,
                            "model": model,
                        },
                    ) from exc

    return _agent_registry[agent_id][1]


def get_agent_resources(agent_id: str) -> AgentInstance:
    """Return the factory's instance object for an already-created agent.

    :func:`get_agent` hands back only the graph, which is all a request needs.
    Anything that has to reach past the graph lives on the instance beside it.

    Args:
        agent_id: The registered agent id.

    Returns:
        The instance object the agent's factory returned with its graph.

    Raises:
        AgentNotFoundError: The agent has not been created yet, which means
            the caller ran before the first request.
    """

    if agent_id not in _agent_registry:
        raise AgentNotFoundError(agent_id, details={"error": "agent has not been created yet"})
    return _agent_registry[agent_id][0]


async def shutdown_agents() -> None:
    """Release long-lived resources held by cached agents.

    An agent factory may hold network clients (an MCP session, an HTTP pool)
    open for the process lifetime. If an agent's instance object has
    ``aclose()`` or ``close()``, it is called here; otherwise nothing happens.
    """

    for agent_id, (instance, _graph) in list(_agent_registry.items()):
        close = getattr(instance, "aclose", None) or getattr(instance, "close", None)
        if close is None:
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
            logger.info("Closed resources for agent %s", agent_id)
        except Exception:
            logger.warning("Failed to close resources for agent %s", agent_id, exc_info=True)

    _agent_registry.clear()
    _agent_creation_locks.clear()


def list_registered_agents() -> list[str]:
    """List every agent id with a registered factory, whether or not it is built.

    Returns:
        The registered agent ids.
    """

    return list(_agent_factories)
