"""Fixtures every core test module may use.

pytest imports this file automatically and makes its fixtures available to
every test in ``tests/``, so nothing here needs importing. Tests that need a
settings variant receive ``settings_factory`` and call it -- importing a
``conftest`` module directly is ambiguous once a nested suite has its own
``conftest.py``.

Four other modules still define their own ``make_settings`` with different
defaults (a DEBUG log level, a real database DSN, populated API keys). Those
differences are what those tests are about, so they were left alone rather
than folded into one helper with four flags.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from juena_core import config as config_module
from juena_core.config import CoreSettings
from juena_core.schema.llm_models import BlabladorModelName
from juena_core.server.identity import local_principal


def _make_settings(**overrides: object) -> CoreSettings:
    values: dict[str, object] = {
        "OPENAI_API_KEY": None,
        "BLABLADOR_API_KEY": None,
        "BLABLADOR_BASE_URL": None,
        "MAX_TOKENS": 10_000,
        "TIMEOUT_SECONDS": 60,
        "MAX_RETRIES": 3,
        "DEFAULT_PROVIDER": "blablador",
        "DEFAULT_MODEL": BlabladorModelName.GPT_OSS.value,
        "OPENAI_AVAILABLE_MODELS": None,
        "BLABLADOR_AVAILABLE_MODELS": None,
        "OPENAI_DEFAULT_MODEL": "gpt-4o-mini",
        "BLABLADOR_DEFAULT_MODEL": BlabladorModelName.GPT_OSS.value,
        "STREAM_TOOL_PAYLOADS": False,
        "DATABASE_URL": None,
        "DATABASE_POOL_MAX_SIZE": 5,
        "SESSION_TTL_HOURS": 8,
        "SESSION_COOKIE_SECURE": False,
        "FALLBACK_PROVIDER": None,
        "EXECUTE_TIMEOUT_SECONDS": 600,
        "ARTIFACT_ROOT": Path("/tmp/juena-core-contract-artifacts"),
        "AUDIT_FILE": Path("/tmp/juena-core-contract-artifacts/audit.jsonl"),
        "LOG_LEVEL": "INFO",
        "LOG_DIR": Path("/tmp/juena-core-contract-logs"),
        "BIND_HOST": "127.0.0.1",
        "API_PUBLISHED": False,
    }
    values.update(overrides)
    return CoreSettings(**values)  # type: ignore[arg-type]


@pytest.fixture
def configured(monkeypatch):
    """Install core settings process-wide for one test.

    Patching the module global rather than each importer's reference: every
    module holds the same ``settings`` function, and that function reads this.
    """

    config = _make_settings()
    monkeypatch.setattr(config_module, "_settings", config)
    return config


@pytest.fixture
def settings_factory():
    """Build a settings variant without importing this conftest module."""

    return _make_settings


@pytest.fixture
def local(configured):
    return local_principal(user_id=uuid4())

