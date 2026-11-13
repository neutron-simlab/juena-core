"""Configuration supplied by the application embedding :mod:`juena_core`.

This module deliberately does not read environment variables or a ``.env``
file. Each application owns that policy and calls :func:`configure` once
during startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from pathlib import Path

__all__ = ["CoreSettings", "configure", "settings"]


@dataclass(frozen=True, slots=True)
class CoreSettings:
    """Values consumed by shared core modules.

    The capitalised names match the application's own settings, so an
    application can fill this field by field rather than keep a second
    configuration system.

    Attributes:
        OPENAI_API_KEY: OpenAI API key; the provider is unavailable without it.
        BLABLADOR_API_KEY: Blablador API key, or an async function that returns
            one for the current request.
        BLABLADOR_BASE_URL: Base URL of the Blablador endpoint.
        MAX_TOKENS: Default maximum tokens per model answer.
        TIMEOUT_SECONDS: Default model request timeout.
        MAX_RETRIES: Default number of retries inside the model client.
        DEFAULT_PROVIDER: Provider used when a request names none.
        DEFAULT_MODEL: Model used when a request names none.
        OPENAI_AVAILABLE_MODELS: Comma-separated OpenAI models to offer; empty
            offers them all.
        BLABLADOR_AVAILABLE_MODELS: Comma-separated Blablador models to offer;
            empty offers them all.
        OPENAI_DEFAULT_MODEL: Default OpenAI model.
        BLABLADOR_DEFAULT_MODEL: Default Blablador model.
        STREAM_TOOL_PAYLOADS: Whether chat messages sent to clients include
            tool-call payloads.
        DATABASE_URL: PostgreSQL connection URL.
        DATABASE_POOL_MAX_SIZE: Largest size of the database connection pool.
        SESSION_TTL_HOURS: How long a sign-in session lasts.
        FALLBACK_PROVIDER: Provider whose default model is the fallback when no
            explicit fallback models are given; empty means no fallback.
        EXECUTE_TIMEOUT_SECONDS: Longest timeout an agent may set for one
            ``execute`` call.
        ARTIFACT_ROOT: Directory where generated files are stored.
        AUDIT_FILE: File that artifact audit records are appended to.
        LOG_LEVEL: Logging level name, such as ``"INFO"``.
        BIND_HOST: Host the API listens on.
        API_PUBLISHED: Whether the API port is reachable from outside its
            container.
    """

    OPENAI_API_KEY: str | None
    # A supplier is evaluated by the asynchronous model client per request.
    # Applications own identity lookup; core never stores personal credentials.
    BLABLADOR_API_KEY: str | Callable[[], Awaitable[str]] | None
    BLABLADOR_BASE_URL: str | None
    MAX_TOKENS: int
    TIMEOUT_SECONDS: int
    MAX_RETRIES: int
    DEFAULT_PROVIDER: str
    DEFAULT_MODEL: str
    OPENAI_AVAILABLE_MODELS: str | None
    BLABLADOR_AVAILABLE_MODELS: str | None
    OPENAI_DEFAULT_MODEL: str
    BLABLADOR_DEFAULT_MODEL: str

    STREAM_TOOL_PAYLOADS: bool

    DATABASE_URL: str | None
    DATABASE_POOL_MAX_SIZE: int

    SESSION_TTL_HOURS: int

    FALLBACK_PROVIDER: str | None
    EXECUTE_TIMEOUT_SECONDS: int

    ARTIFACT_ROOT: Path
    AUDIT_FILE: Path

    LOG_LEVEL: str

    BIND_HOST: str
    API_PUBLISHED: bool


_settings: CoreSettings | None = None


def configure(settings_: CoreSettings) -> None:
    """Install the application's immutable core settings.

    Applications call this once at startup. Repeating the same configuration
    is harmless, but replacing it would leave cached models and other
    process-level resources paired with stale values.

    Args:
        settings_: The settings to install.

    Raises:
        RuntimeError: Different settings were already installed.
    """

    global _settings
    if _settings is not None and _settings != settings_:
        raise RuntimeError("juena_core.configure() has already been called")
    _settings = settings_


def settings() -> CoreSettings:
    """Return configured settings, failing clearly before application startup.

    Returns:
        The settings passed to :func:`configure`.

    Raises:
        RuntimeError: :func:`configure` has not been called yet.
    """

    if _settings is None:
        raise RuntimeError("juena_core.configure() has not been called")
    return _settings
