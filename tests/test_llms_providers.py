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


def test_default_model_comes_from_application_settings(monkeypatch) -> None:
    config = make_settings(
        BLABLADOR_DEFAULT_MODEL=BlabladorModelName.QWEN38_FLASH_NEXT.value
    )
    monkeypatch.setattr(providers, "settings", lambda: config)

    assert providers.get_default_model("blablador") == (
        "02 - Qwen3.8-Flash-Next-NVFP4, general purpose large model"
    )


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
        "stream_usage": True,
    }


def test_juwels_qwen_profile_reserves_output_within_served_context(monkeypatch) -> None:
    monkeypatch.setattr(providers, "settings", make_settings)

    kwargs = providers._model_kwargs(
        "blablador",
        "90 - Qwen 3.8-Flash-Next-125B-AWQ-W4A16 on Juwels Booster - "
        "An excellent model from September 2026",
        0.0,
        {"max_tokens": 32_768},
    )

    assert kwargs["max_tokens"] == 32_768
    assert kwargs["profile"] == {"max_input_tokens": 907_232}


def test_rate_limited_model_shares_one_pacer_across_clients(monkeypatch) -> None:
    """BlaBlaDor refuses more than about 30 HZDR requests a minute. Every client
    of that model must wait on the same limiter, whatever its other settings."""
    from langchain_core.rate_limiters import InMemoryRateLimiter

    monkeypatch.setattr(providers, "settings", make_settings)
    providers._build_cached.cache_clear()
    hzdr = BlabladorModelName.DEEPSEEK_V41_FLASH_HZDR.value
    try:
        specialist = providers.build_chat_model(
            provider="blablador", model=hzdr, max_tokens=32_768, streaming=True, max_retries=0
        )
        supervisor = providers.build_chat_model(provider="blablador", model=hzdr)
        juwels = providers.build_chat_model(
            provider="blablador", model=BlabladorModelName.DEEPSEEK_V41_FLASH_JUWELS.value
        )
    finally:
        providers._build_cached.cache_clear()

    assert specialist is not supervisor
    assert isinstance(specialist.rate_limiter, InMemoryRateLimiter)
    assert specialist.rate_limiter is supervisor.rate_limiter
    assert specialist.rate_limiter.requests_per_second == 0.45
    assert juwels.rate_limiter is None
    assert specialist.profile["max_input_tokens"] == 900_000 - 32_768


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


def test_same_model_ignores_settings_but_not_model_or_endpoint(monkeypatch) -> None:
    config = make_settings()
    monkeypatch.setattr(providers, "settings", lambda: config)
    qwen = BlabladorModelName.QWEN38_FLASH_NEXT_JUWELS.value

    primary = providers._build("blablador", qwen, 0.0, {"max_tokens": 32_768, "streaming": True})
    same_with_other_settings = providers._build("blablador", qwen, 0.0, {})
    other_model = providers._build("blablador", BlabladorModelName.GPT_OSS.value, 0.0, {})
    other_endpoint = providers._build("openai", qwen, 0.0, {})

    assert providers.same_model(primary, same_with_other_settings)
    assert not providers.same_model(primary, other_model)
    assert not providers.same_model(primary, other_endpoint)
    assert not providers.same_model(object(), object())


@pytest.mark.asyncio
async def test_streamed_specialist_call_keeps_report_tool_call_and_usage(monkeypatch) -> None:
    """Specialists stream; the report tool call and token usage must survive it."""
    import json

    import httpx
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy
    from langchain.chat_models import init_chat_model
    from langchain_core.tools import tool

    from juena_core.schema.agents import SpecialistReport

    config = make_settings()
    monkeypatch.setattr(providers, "settings", lambda: config)
    requests: list[dict] = []

    def chunk(**fields) -> str:
        body = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", **fields}
        return "data: " + json.dumps(body) + "\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        arguments = json.dumps({"status": "completed", "finding": "The value is 17."})
        stream = (
            chunk(choices=[{"index": 0, "delta": {"role": "assistant", "tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "SpecialistReport", "arguments": ""}}]}, "finish_reason": None}])
            + chunk(choices=[{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": arguments}}]}, "finish_reason": None}])
            + chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])
            + chunk(choices=[], usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18})
            + "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})

    kwargs = providers._model_kwargs(
        "blablador", BlabladorModelName.QWEN38_FLASH_NEXT_JUWELS.value, 0.0,
        {"streaming": True, "max_retries": 0},
    )
    model = init_chat_model(
        BlabladorModelName.QWEN38_FLASH_NEXT_JUWELS.value,
        **kwargs,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    @tool
    def lookup_value() -> str:
        """Return the stored value."""
        return "17"

    agent = create_agent(model=model, tools=[lookup_value], response_format=ToolStrategy(SpecialistReport))
    result = await agent.ainvoke({"messages": [("user", "What is the value?")]})

    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert result["structured_response"].finding == "The value is 17."
    model_reply = next(m for m in result["messages"] if getattr(m, "tool_calls", None))
    assert model_reply.usage_metadata["input_tokens"] == 11
