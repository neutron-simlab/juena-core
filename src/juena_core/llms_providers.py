"""LLM construction for the supported providers.

Both providers currently expose the OpenAI Chat Completions API. Blablador
therefore uses LangChain's OpenAI provider with a custom base URL.
"""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from juena_core.config import settings
from juena_core.schema.llm_models import (
    BLABLADOR_MAX_MODEL_LEN,
    Provider,
    get_default_model_for_provider,
    get_models_for_provider,
)

__all__ = [
    "get_available_providers",
    "build_chat_model",
    "get_available_models",
    "get_default_model",
    "format_model_name",
]

_PROVIDERS: tuple[str, ...] = (Provider.OPENAI.value, Provider.BLABLADOR.value)


def _is_available(provider: str) -> bool:
    """Return whether *provider* has the credentials it requires."""

    config = settings()
    if provider == Provider.OPENAI.value:
        return bool((config.OPENAI_API_KEY or "").strip())
    if provider == Provider.BLABLADOR.value:
        return bool(
            (config.BLABLADOR_API_KEY or "").strip()
            and (config.BLABLADOR_BASE_URL or "").strip()
        )
    return False


def get_available_providers() -> dict[str, bool]:
    """Return availability for every provider this build can construct."""

    return {name: _is_available(name) for name in _PROVIDERS}


def _model_kwargs(
    provider: str,
    model: str,
    temperature: float,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Build the ``init_chat_model`` keyword arguments for a provider."""

    config = settings()
    kwargs: dict[str, Any] = {
        "model_provider": Provider.OPENAI.value,
        "temperature": temperature,
        "max_tokens": overrides.get("max_tokens", config.MAX_TOKENS),
        "timeout": overrides.get("timeout", config.TIMEOUT_SECONDS),
        "max_retries": overrides.get("max_retries", config.MAX_RETRIES),
    }

    if provider == Provider.BLABLADOR.value:
        kwargs["api_key"] = config.BLABLADOR_API_KEY
        kwargs["base_url"] = config.BLABLADOR_BASE_URL
        context_window = BLABLADOR_MAX_MODEL_LEN.get(model)
        if context_window is not None:
            kwargs["profile"] = {
                "max_input_tokens": context_window - kwargs["max_tokens"]
            }
    else:
        kwargs["api_key"] = config.OPENAI_API_KEY

    if "streaming" in overrides:
        kwargs["streaming"] = overrides["streaming"]

    return kwargs


def _build(
    provider: str,
    model: str,
    temperature: float,
    overrides: dict[str, Any],
) -> BaseChatModel:
    return init_chat_model(
        model,
        **_model_kwargs(provider, model, temperature, overrides),
    )


@lru_cache(maxsize=32)
def _build_cached(
    provider: str,
    model: str,
    temperature: float,
    override_items: tuple[tuple[str, Any], ...],
) -> BaseChatModel:
    """Build once per configuration so model-owned HTTP pools are reused."""

    return _build(provider, model, temperature, dict(override_items))


def build_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **overrides: Any,
) -> BaseChatModel:
    """Create a configured chat model for a supported provider."""

    config = settings()
    provider = (provider or config.DEFAULT_PROVIDER).lower()
    model = model or config.DEFAULT_MODEL

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {', '.join(_PROVIDERS)}"
        )

    override_items = tuple(sorted(overrides.items()))
    try:
        hash(override_items)
    except TypeError:
        return _build(provider, model, temperature, overrides)
    return _build_cached(provider, model, temperature, override_items)


def get_available_models(provider: str) -> list[str]:
    """Return models allowed for *provider*, applying its configured filter."""

    config = settings()
    provider = provider.lower()

    try:
        all_models = get_models_for_provider(Provider(provider))
    except ValueError:
        return []

    configured_filter = None
    if provider == Provider.OPENAI.value:
        configured_filter = config.OPENAI_AVAILABLE_MODELS
    elif provider == Provider.BLABLADOR.value:
        configured_filter = config.BLABLADOR_AVAILABLE_MODELS

    if configured_filter:
        # Some provider-owned model identifiers contain commas themselves.
        # Match complete known identifiers at list boundaries rather than
        # splitting and corrupting those identifiers.
        configured_filter = configured_filter.strip()
        filtered = [
            item
            for item in all_models
            if re.search(
                rf"(?:^|,\s*){re.escape(item)}(?:\s*,|$)",
                configured_filter,
            )
        ]
        return filtered if filtered else all_models

    return all_models


def get_default_model(provider: str) -> str:
    """Return the built-in default model, or ``""`` for an unknown provider."""

    try:
        return get_default_model_for_provider(Provider(provider.lower()))
    except ValueError:
        return ""


def format_model_name(provider: str, model: str) -> str:
    """Format a provider model identifier for display."""

    if provider.lower() == Provider.BLABLADOR.value:
        from juena_core.schema.llm_models import get_blablador_model_display_name

        return get_blablador_model_display_name(model)
    return model
