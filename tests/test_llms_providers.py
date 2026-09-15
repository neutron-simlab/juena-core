"""CP1 tests for provider discovery and model construction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import juena_core.llms_providers as providers
from juena_core.config import CoreSettings
from juena_core.schema.llm_models import BlabladorModelName


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
        "BLABLADOR_DEFAULT_MODEL": BlabladorModelName.GPT_OSS.value,
        "STREAM_TOOL_PAYLOADS": False,
        "DATABASE_URL": None,
        "DATABASE_POOL_MAX_SIZE": 20,
        "SESSION_TTL_HOURS": 8,
        "SESSION_COOKIE_SECURE": False,
        "FALLBACK_PROVIDER": None,
        "EXECUTE_TIMEOUT_SECONDS": 600,
        "ARTIFACT_ROOT": Path("/tmp/artifacts"),
        "AUDIT_FILE": Path("/tmp/audit.jsonl"),
        "LOG_LEVEL": "INFO",
        "LOG_DIR": Path("/tmp/logs"),
        "BIND_HOST": "127.0.0.1",
        "API_PUBLISHED": False,
    }
    values.update(overrides)
    return CoreSettings(**values)  # type: ignore[arg-type]


def test_provider_availability_requires_complete_credentials(monkeypatch) -> None:
    config = make_settings(OPENAI_API_KEY=" ", BLABLADOR_BASE_URL=None)
    monkeypatch.setattr(providers, "settings", lambda: config)

    assert providers.get_available_providers() == {
        "openai": False,
        "blablador": False,
    }


def test_blablador_kwargs_include_endpoint_and_measured_profile(monkeypatch) -> None:
    config = make_settings()
    monkeypatch.setattr(providers, "settings", lambda: config)

    kwargs = providers._model_kwargs(
        "blablador",
        BlabladorModelName.GPT_OSS.value,
        0.25,
        {"max_tokens": 2_000, "streaming": True},
    )

    assert kwargs == {
        "model_provider": "openai",
        "temperature": 0.25,
        "max_tokens": 2_000,
        "timeout": 60,
        "max_retries": 3,
        "api_key": "blablador-key",
        "base_url": "https://blablador.example/v1",
        "profile": {"max_input_tokens": 131_072 - 2_000},
        "streaming": True,
    }


def test_build_chat_model_reuses_hashable_constructions(monkeypatch) -> None:
    config = make_settings()
    built: list[tuple[object, ...]] = []

    def fake_build(
        provider: str,
        model: str,
        temperature: float,
        overrides: dict,
    ) -> object:
        built.append((provider, model, temperature, overrides))
        return SimpleNamespace(provider=provider, model=model)

    monkeypatch.setattr(providers, "settings", lambda: config)
    monkeypatch.setattr(providers, "_build", fake_build)
    providers._build_cached.cache_clear()

    first = providers.build_chat_model(streaming=True)
    second = providers.build_chat_model(streaming=True)

    assert first is second
    assert len(built) == 1
    providers._build_cached.cache_clear()


def test_unhashable_override_skips_the_model_cache(monkeypatch) -> None:
    config = make_settings()
    built: list[dict] = []
    monkeypatch.setattr(providers, "settings", lambda: config)
    monkeypatch.setattr(
        providers,
        "_build",
        lambda _provider, _model, _temperature, overrides: built.append(overrides)
        or object(),
    )

    providers.build_chat_model(model_kwargs={"seed": 1})
    providers.build_chat_model(model_kwargs={"seed": 1})

    assert len(built) == 2


def test_available_model_filter_keeps_only_known_models(monkeypatch) -> None:
    selected = BlabladorModelName.MINIMAX_M27.value
    config = make_settings(BLABLADOR_AVAILABLE_MODELS=f"unknown, {selected}")
    monkeypatch.setattr(providers, "settings", lambda: config)

    assert providers.get_available_models("BLABLADOR") == [selected]
    assert providers.get_available_models("not-a-provider") == []


def test_unknown_provider_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(providers, "settings", make_settings)

    with pytest.raises(ValueError, match="Unknown provider: local"):
        providers.build_chat_model(provider="local")
