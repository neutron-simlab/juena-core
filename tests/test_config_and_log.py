"""CP1 tests for explicit configuration and import-safe logging."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
import logging
from pathlib import Path

import pytest

import juena_core
import juena_core.config as config_module
from juena_core.config import CoreSettings
from juena_core.log import _get_log_level, setup_logger


EXPECTED_SETTINGS = {
    "OPENAI_API_KEY",
    "BLABLADOR_API_KEY",
    "BLABLADOR_BASE_URL",
    "MAX_TOKENS",
    "TIMEOUT_SECONDS",
    "MAX_RETRIES",
    "DEFAULT_PROVIDER",
    "DEFAULT_MODEL",
    "OPENAI_AVAILABLE_MODELS",
    "BLABLADOR_AVAILABLE_MODELS",
    "OPENAI_DEFAULT_MODEL",
    "BLABLADOR_DEFAULT_MODEL",
    "STREAM_TOOL_PAYLOADS",
    "DATABASE_URL",
    "DATABASE_POOL_MAX_SIZE",
    "SESSION_TTL_HOURS",
    "SESSION_COOKIE_SECURE",
    "FALLBACK_PROVIDER",
    "EXECUTE_TIMEOUT_SECONDS",
    "ARTIFACT_ROOT",
    "AUDIT_FILE",
    "LOG_LEVEL",
    "LOG_DIR",
    "BIND_HOST",
    "API_PUBLISHED",
}


def make_settings(**overrides: object) -> CoreSettings:
    values: dict[str, object] = {
        "OPENAI_API_KEY": "openai-key",
        "BLABLADOR_API_KEY": "blablador-key",
        "BLABLADOR_BASE_URL": "https://blablador.example/v1",
        "MAX_TOKENS": 10_000,
        "TIMEOUT_SECONDS": 60,
        "MAX_RETRIES": 3,
        "DEFAULT_PROVIDER": "openai",
        "DEFAULT_MODEL": "gpt-4o-mini",
        "OPENAI_AVAILABLE_MODELS": None,
        "BLABLADOR_AVAILABLE_MODELS": None,
        "OPENAI_DEFAULT_MODEL": "gpt-4o-mini",
        "BLABLADOR_DEFAULT_MODEL": "blablador-default",
        "STREAM_TOOL_PAYLOADS": False,
        "DATABASE_URL": None,
        "DATABASE_POOL_MAX_SIZE": 20,
        "SESSION_TTL_HOURS": 8,
        "SESSION_COOKIE_SECURE": False,
        "FALLBACK_PROVIDER": None,
        "EXECUTE_TIMEOUT_SECONDS": 600,
        "ARTIFACT_ROOT": Path("/tmp/artifacts"),
        "AUDIT_FILE": Path("/tmp/audit.jsonl"),
        "LOG_LEVEL": "DEBUG",
        "LOG_DIR": Path("/tmp/logs"),
        "BIND_HOST": "127.0.0.1",
        "API_PUBLISHED": False,
    }
    values.update(overrides)
    return CoreSettings(**values)  # type: ignore[arg-type]


def test_core_settings_has_the_complete_shared_boundary() -> None:
    assert {field.name for field in fields(CoreSettings)} == EXPECTED_SETTINGS


def test_configure_installs_an_immutable_settings_object(monkeypatch) -> None:
    configured = make_settings()
    monkeypatch.setattr(config_module, "_settings", None)

    juena_core.configure(configured)

    assert juena_core.settings() is configured
    with pytest.raises(FrozenInstanceError):
        configured.LOG_LEVEL = "ERROR"  # type: ignore[misc]


def test_configure_refuses_a_different_process_configuration(monkeypatch) -> None:
    configured = make_settings()
    monkeypatch.setattr(config_module, "_settings", None)

    juena_core.configure(configured)
    juena_core.configure(make_settings())

    with pytest.raises(RuntimeError, match="has already been called"):
        juena_core.configure(make_settings(LOG_LEVEL="ERROR"))


def test_settings_fails_clearly_before_configuration(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "_settings", None)

    with pytest.raises(
        RuntimeError,
        match=r"juena_core\.configure\(\) has not been called",
    ):
        juena_core.settings()


def test_config_module_has_no_environment_or_dotenv_dependency() -> None:
    path = Path(config_module.__file__ or "")
    tree = ast.parse(path.read_text(), filename=str(path))
    imported_roots: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called_names.add(node.func.id)

    assert "dotenv" not in imported_roots
    assert "os" not in imported_roots
    assert "getenv" not in called_names


def test_logging_defaults_to_info_before_configuration(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "_settings", None)
    assert _get_log_level() == logging.INFO


def test_logging_reads_configured_level_at_call_time(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "_settings", make_settings(LOG_LEVEL="warning"))
    assert _get_log_level() == logging.WARNING


def test_setup_logger_does_not_duplicate_handlers() -> None:
    logger = logging.Logger("cp1-test")

    setup_logger(logger, "ERROR")
    setup_logger(logger, "DEBUG")

    assert len(logger.handlers) == 1
    assert logger.level == logging.ERROR
    assert logger.handlers[0].level == logging.ERROR
    assert logger.propagate is False
