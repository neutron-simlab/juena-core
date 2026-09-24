"""Runtime model selection middleware for LangChain/LangGraph agents.

The UI owns provider/model selection. The server passes that selection through
LangChain runtime context on every invocation, and this middleware swaps the
request model at call time so compiled agent graphs do not need to be rebuilt.

``RuntimeModelContext`` also carries ``thread_id`` and ``user_id`` — half the
identity seam. ``user_id`` is the string form of ``Principal.id``, and it is
what a tool reads to scope a per-user store or artifact namespace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from juena_core.llms_providers import build_chat_model
from juena_core.log import get_logger

logger = get_logger(__name__)

__all__ = ["RuntimeModelContext", "RuntimeModelMiddleware"]


@dataclass(frozen=True, slots=True)
class RuntimeModelContext:
    """Per-request runtime context: which model to use, and whose run this is.

    ``run_id`` identifies one invocation of the graph, and it travels here
    rather than in the ``RunnableConfig`` because a subgraph does not inherit
    the parent's config ``run_id``. LangGraph populates
    ``Runtime.execution_info.run_id`` from the config of the graph actually
    running, so inside a specialist subagent it is ``None`` -- while the
    runtime *context* is passed down unchanged. Execution evidence is written
    inside a specialist and read outside it, so the id both halves agree on has
    to be the one that crosses that boundary.
    """

    provider: str
    model: str
    thread_id: str | None = None
    user_id: str | None = None
    run_id: str | None = None


def _provider_model_from_context(context: Any) -> tuple[str | None, str | None]:
    """Extract provider/model from LangChain runtime context.

    Handles both a dataclass context and a plain dict, because an application
    may pass either to ``astream``.
    """

    if context is None:
        return None, None
    if isinstance(context, dict):
        provider = context.get("provider")
        model = context.get("model")
    else:
        provider = getattr(context, "provider", None)
        model = getattr(context, "model", None)
    return str(provider) if provider else None, str(model) if model else None


def _get_provider_model_from_request(request: ModelRequest) -> tuple[str | None, str | None]:
    """Resolve provider/model for the current model call."""

    return _provider_model_from_context(getattr(request.runtime, "context", None))


class RuntimeModelMiddleware(AgentMiddleware):
    """Swap the active LLM per request using runtime context."""

    def _override_request_model(self, request: ModelRequest) -> ModelRequest:
        provider, model = _get_provider_model_from_request(request)
        if not provider or not model:
            return request

        try:
            # build_chat_model caches on (provider, model, temperature, overrides),
            # so this does not rebuild a connection pool on every model call.
            model_kwargs: dict[str, Any] = {
                "provider": provider,
                "model": model,
                "temperature": 0.0,
            }
            # Explicit ``streaming=False`` sets LangChain's internal
            # ``_streaming_disabled`` flag. That prevents LangGraph's messages mode
            # from attaching its streaming callback after a runtime model swap. True
            # is meaningful and must be preserved; false means leave the option
            # unset, as the original production model does.
            if bool(getattr(request.model, "streaming", False)):
                model_kwargs["streaming"] = True
            llm = build_chat_model(**model_kwargs)
        except Exception:
            logger.exception(
                "Failed to create runtime LLM for provider=%s model=%s", provider, model
            )
            raise

        logger.debug("Using runtime LLM provider=%s model=%s", provider, model)
        return request.override(model=llm)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Select the model for the current sync request."""

        return handler(self._override_request_model(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        """Select the model for the current async request."""

        return await handler(self._override_request_model(request))
