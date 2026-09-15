"""Configuration boundary for the optional sandbox runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SandboxRuntimeSettings",
    "configure_sandbox",
    "sandbox_settings",
    "sandbox_enabled",
]


@dataclass(frozen=True, slots=True)
class SandboxRuntimeSettings:
    """Application-side settings for job submission and workspace staging.

    Worker-only limits are read by ``SandboxWorkerSettings`` in ``executor``.
    Values repeated here must match the worker deployment; the API uses them
    for admission, timeouts, and the human approval card.
    """

    enabled: bool
    identity_secret: str | None
    workspace_root: Path
    concurrency: int = 4
    execution_timeout_seconds: int = 600
    max_output_bytes: int = 100_000
    workspace_limit_bytes: int = 5 * 1024**3
    idle_ttl_seconds: int = 24 * 3600
    cpu_limit: str = "2"
    memory_limit: str = "4g"


_sandbox_settings: SandboxRuntimeSettings | None = None


def _validate(value: SandboxRuntimeSettings) -> None:
    if value.concurrency < 1:
        raise ValueError("sandbox concurrency must be at least 1")
    if value.execution_timeout_seconds <= 0:
        raise ValueError("sandbox execution timeout must be positive")
    if value.max_output_bytes <= 0:
        raise ValueError("sandbox max output bytes must be positive")
    if value.workspace_limit_bytes <= 0:
        raise ValueError("sandbox workspace limit must be positive")
    if value.idle_ttl_seconds <= 0:
        raise ValueError("sandbox idle TTL must be positive")
    if value.enabled and len(value.identity_secret or "") < 32:
        raise ValueError(
            "sandbox identity secret must contain at least 32 characters when enabled"
        )


def configure_sandbox(settings_: SandboxRuntimeSettings) -> None:
    """Install sandbox settings before constructing the FastAPI application.

    Importing the optional ORM model here attaches ``sandbox_jobs`` to core's
    metadata before ``create_app`` runs ``create_all``. An application that
    never enables this feature never imports the model or creates the table.
    """

    _validate(settings_)
    global _sandbox_settings
    if _sandbox_settings is not None and _sandbox_settings != settings_:
        raise RuntimeError("configure_sandbox() has already been called")
    _sandbox_settings = settings_
    if settings_.enabled:
        from juena_core.sandbox import models as _models  # noqa: F401


def sandbox_settings() -> SandboxRuntimeSettings:
    """Return configured settings, failing clearly when execution was not wired."""

    if _sandbox_settings is None:
        raise RuntimeError("configure_sandbox() has not been called")
    return _sandbox_settings


def sandbox_enabled() -> bool:
    """Return whether an explicitly configured sandbox is enabled."""

    return bool(_sandbox_settings is not None and _sandbox_settings.enabled)
