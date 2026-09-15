"""The optional sandbox must remain opt-in and independently configured."""

from __future__ import annotations

import subprocess
import sys

import pytest

from juena_core.sandbox import config as sandbox_config
from juena_core.sandbox.config import (
    SandboxRuntimeSettings,
    configure_sandbox,
    sandbox_enabled,
    sandbox_settings,
)


def _settings(tmp_path, **overrides) -> SandboxRuntimeSettings:
    values = {
        "enabled": True,
        "identity_secret": "s" * 32,
        "workspace_root": tmp_path / "workspaces",
    }
    values.update(overrides)
    return SandboxRuntimeSettings(**values)


def test_unconfigured_sandbox_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(sandbox_config, "_sandbox_settings", None)

    assert sandbox_enabled() is False
    with pytest.raises(RuntimeError, match=r"configure_sandbox\(\)"):
        sandbox_settings()


def test_base_import_does_not_load_optional_podman_client() -> None:
    code = "import sys; import juena_core; assert 'podman' not in sys.modules"
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_configuration_is_immutable_and_validated(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sandbox_config, "_sandbox_settings", None)
    configured = _settings(tmp_path, enabled=False, identity_secret=None)

    configure_sandbox(configured)
    configure_sandbox(configured)

    with pytest.raises(RuntimeError, match="already been called"):
        configure_sandbox(_settings(tmp_path))
    monkeypatch.setattr(sandbox_config, "_sandbox_settings", None)
    with pytest.raises(ValueError, match="at least 32 characters"):
        configure_sandbox(_settings(tmp_path, identity_secret="short"))


def test_enabling_sandbox_registers_only_its_optional_table(tmp_path) -> None:
    code = f"""
from pathlib import Path
from juena_core.sandbox.config import SandboxRuntimeSettings, configure_sandbox
from juena_core.server.database.models import Base

assert set(Base.metadata.tables) == {{"users", "auth_sessions", "chats"}}
configure_sandbox(SandboxRuntimeSettings(
    enabled=True,
    identity_secret="{'s' * 32}",
    workspace_root=Path({str(tmp_path / 'workspaces')!r}),
))
assert set(Base.metadata.tables) == {{
    "users", "auth_sessions", "chats", "sandbox_jobs"
}}
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
