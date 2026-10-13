"""Let the application attach its tracer to every agent run.

LangSmith needs nothing from here: LangChain switches it on from environment
variables. A tracer that works as a callback handler, such as Langfuse, has to
travel in the run's ``RunnableConfig`` instead. Core builds that config, so the
application hands core one function that says what each run should carry.
Core never imports a tracing vendor itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

__all__ = ["RunTracing", "TracingHook", "configure_tracing", "with_tracing"]


@dataclass(frozen=True, slots=True)
class RunTracing:
    """What one run carries for the application's tracer."""

    callbacks: Sequence[BaseCallbackHandler] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


#: Called as ``hook(thread_id=..., user_id=...)`` once per run.
TracingHook = Callable[..., RunTracing]

_hook: TracingHook | None = None


def configure_tracing(hook: TracingHook) -> None:
    """Install the application's tracing hook, once, at startup.

    Same rule as ``configure()``: repeating the same hook is harmless, but a
    second, different hook would silently change what earlier runs reported.
    """

    global _hook
    if _hook is not None and _hook is not hook:
        raise RuntimeError("configure_tracing() has already been called")
    _hook = hook


def with_tracing(
    config: RunnableConfig, *, thread_id: str, user_id: str
) -> RunnableConfig:
    """Return ``config`` plus the hook's callbacks and metadata for this run.

    A plain dict merge, not LangChain's ``merge_configs``: that calls
    ``ensure_config``, which also copies in the config of whatever run is
    active in the caller's context -- exactly what a detached run must not
    inherit.
    """

    if _hook is None:
        return config
    tracing = _hook(thread_id=thread_id, user_id=user_id)
    return {
        **config,
        "callbacks": [*(config.get("callbacks") or []), *tracing.callbacks],
        "metadata": {**(config.get("metadata") or {}), **tracing.metadata},
    }
