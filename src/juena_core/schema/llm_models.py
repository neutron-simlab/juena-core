"""Supported providers and their shared model catalogue."""

from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

__all__ = [
    "Provider",
    "OpenAIModelName",
    "BlabladorModelName",
    "AllModelEnum",
    "BLABLADOR_MAX_MODEL_LEN",
    "BLABLADOR_MODEL_DISPLAY_NAMES",
    "BLABLADOR_REQUESTS_PER_SECOND",
    "get_blablador_model_display_name",
    "PROVIDER_MODELS",
    "get_models_for_provider",
    "get_default_model_for_provider",
]


class Provider(StrEnum):
    """LLM providers supported by the shared runtime."""

    OPENAI = "openai"
    BLABLADOR = "blablador"


class OpenAIModelName(StrEnum):
    """OpenAI chat models intentionally offered by this deployment."""

    GPT_4O_MINI = "gpt-4o-mini"


class BlabladorModelName(StrEnum):
    """Curated models served by the OpenAI-compatible Blablador endpoint."""

    GPT_OSS = "01 - GPT-OSS-120b - an open model released by OpenAI in August 2025"
    QWEN38_FLASH_NEXT_JUWELS = (
        "90 - Qwen 3.8-Flash-Next-125B-AWQ-W4A16 on Juwels Booster - "
        "An excellent model from September 2026"
    )
    DEEPSEEK_V41_FLASH_JUWELS = "15 - DeepSeek-V4.1-Flash on JUWELS Booster"
    DEEPSEEK_V41_FLASH_HZDR = "deepseek-v4.1-flash-at-hzdr"


# Context window per model, as the provider's /v1/models endpoint reports it.
# LangChain has no profile entry for these custom identifiers.
BLABLADOR_MAX_MODEL_LEN: dict[str, int] = {
    # JUWELS /v1/models reports 940k, not 1M.
    BlabladorModelName.QWEN38_FLASH_NEXT_JUWELS.value: 940_000,
    BlabladorModelName.GPT_OSS.value: 131_072,
    BlabladorModelName.DEEPSEEK_V41_FLASH_JUWELS.value: 1_048_576,
    # /v1/models reports no length for it; a 900,043-token prompt was
    # accepted, so this is a verified lower bound.
    BlabladorModelName.DEEPSEEK_V41_FLASH_HZDR.value: 900_000,
}

BLABLADOR_MODEL_DISPLAY_NAMES: dict[str, str] = {
    BlabladorModelName.QWEN38_FLASH_NEXT_JUWELS.value: "Qwen3.8-Flash-Next (JUWELS, 940k)",
    BlabladorModelName.GPT_OSS.value: "GPT-OSS-120b",
    BlabladorModelName.DEEPSEEK_V41_FLASH_JUWELS.value: "DeepSeek-V4.1-Flash (JUWELS)",
    BlabladorModelName.DEEPSEEK_V41_FLASH_HZDR.value: "DeepSeek-V4.1-Flash (HZDR)",
}

# BlaBlaDor answers 429 (too many requests) after about 30 requests a minute
# to these models (other models were unaffected), so clients pace their
# requests below that.
BLABLADOR_REQUESTS_PER_SECOND: dict[str, float] = {
    BlabladorModelName.DEEPSEEK_V41_FLASH_HZDR.value: 0.45,
}


def get_blablador_model_display_name(model_id: str) -> str:
    """Return the UI label for a Blablador model identifier.

    Args:
        model_id: The model identifier.

    Returns:
        The label, or ``model_id`` itself when it has none.
    """

    return BLABLADOR_MODEL_DISPLAY_NAMES.get(model_id, model_id)


AllModelEnum: TypeAlias = OpenAIModelName | BlabladorModelName

PROVIDER_MODELS: dict[Provider, list[str]] = {
    Provider.OPENAI: [model.value for model in OpenAIModelName],
    Provider.BLABLADOR: [model.value for model in BlabladorModelName],
}


def get_models_for_provider(provider: Provider) -> list[str]:
    """Return the models offered for *provider*.

    Args:
        provider: The provider.

    Returns:
        Its model identifiers, or an empty list.
    """

    return PROVIDER_MODELS.get(provider, [])


def get_default_model_for_provider(provider: Provider) -> str:
    """Return the built-in default model for *provider*.

    Args:
        provider: The provider.

    Returns:
        The default model identifier, or ``""`` for an unknown provider.
    """

    defaults = {
        Provider.OPENAI: OpenAIModelName.GPT_4O_MINI.value,
        Provider.BLABLADOR: BlabladorModelName.QWEN38_FLASH_NEXT_JUWELS.value,
    }
    return defaults.get(provider, "")
