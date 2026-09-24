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
    QWEN38_FLASH_NEXT = (
        "02 - Qwen3.8-Flash-Next-NVFP4, general purpose large model"
    )
    MIMO_V26_PRO = "80 - MiMo-V2.6-Pro-RL on Juwels Booster"
    MINIMAX_M27 = "01 - MiniMax-M2.7 - our best model as of April, 2026"
    QWEN35_122B = "02 - Qwen3.5-122B-A10B-FP8, general purpose large model"


# Measured from the provider's /v1/models endpoint; the selected Qwen and MiMo
# entries were refreshed on 2026-09-23. LangChain has no profile entry for
# these custom identifiers.
BLABLADOR_MAX_MODEL_LEN: dict[str, int] = {
    BlabladorModelName.QWEN38_FLASH_NEXT.value: 262_144,
    BlabladorModelName.MIMO_V26_PRO.value: 1_048_576,
    BlabladorModelName.GPT_OSS.value: 131_072,
    BlabladorModelName.MINIMAX_M27.value: 131_072,
    BlabladorModelName.QWEN35_122B.value: 262_144,
}

BLABLADOR_MODEL_DISPLAY_NAMES: dict[str, str] = {
    BlabladorModelName.QWEN38_FLASH_NEXT.value: "Qwen3.8-Flash-Next",
    BlabladorModelName.MIMO_V26_PRO.value: "MiMo-V2.6-Pro",
    BlabladorModelName.GPT_OSS.value: "GPT-OSS-120b",
    BlabladorModelName.MINIMAX_M27.value: "MiniMax-M2.7",
    BlabladorModelName.QWEN35_122B.value: "QWEN3.5-122B",
}


def get_blablador_model_display_name(model_id: str) -> str:
    """Return the UI label for a Blablador model identifier."""

    return BLABLADOR_MODEL_DISPLAY_NAMES.get(model_id, model_id)


AllModelEnum: TypeAlias = OpenAIModelName | BlabladorModelName

PROVIDER_MODELS: dict[Provider, list[str]] = {
    Provider.OPENAI: [model.value for model in OpenAIModelName],
    Provider.BLABLADOR: [model.value for model in BlabladorModelName],
}


def get_models_for_provider(provider: Provider) -> list[str]:
    """Return the models offered for *provider*."""

    return PROVIDER_MODELS.get(provider, [])


def get_default_model_for_provider(provider: Provider) -> str:
    """Return the built-in default model for *provider*."""

    defaults = {
        Provider.OPENAI: OpenAIModelName.GPT_4O_MINI.value,
        Provider.BLABLADOR: BlabladorModelName.GPT_OSS.value,
    }
    return defaults.get(provider, "")
