"""Configuration supplied by the application embedding :mod:`juena_core`.

This module deliberately does not read environment variables or a ``.env``
file. Each application owns that policy and calls :func:`configure` once
during startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["CoreSettings", "configure", "settings"]


@dataclass(frozen=True, slots=True)
class CoreSettings:
    """Values consumed by shared core modules.

    The capitalised names intentionally match the existing application
    settings, making the eventual cutover a mechanical hand-off rather than a
    second configuration system.
    """

    OPENAI_API_KEY: str | None
    BLABLADOR_API_KEY: str | None
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
    SESSION_COOKIE_SECURE: bool

    FALLBACK_PROVIDER: str | None
    EXECUTE_TIMEOUT_SECONDS: int

    ARTIFACT_ROOT: Path
    AUDIT_FILE: Path

    LOG_LEVEL: str
    LOG_DIR: Path

    BIND_HOST: str
    API_PUBLISHED: bool


_settings: CoreSettings | None = None


def configure(settings_: CoreSettings) -> None:
    """Install the application's immutable core settings.

    Applications call this once at startup. Repeating the same configuration
    is harmless, but replacing it would leave cached models and other
    process-level resources paired with stale values.
    """

    global _settings
    if _settings is not None and _settings != settings_:
        raise RuntimeError("juena_core.configure() has already been called")
    _settings = settings_


def settings() -> CoreSettings:
    """Return configured settings, failing clearly before application startup."""

    if _settings is None:
        raise RuntimeError("juena_core.configure() has not been called")
    return _settings
