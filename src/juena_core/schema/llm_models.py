"""Stub for 01/CP1. Ported from ``juena/schema/llm_models.py``. Arguably
application policy — *which* models are offered — but both apps offer the
same two providers against the same institute endpoint (00-BOUNDARY.md,
*Moves whole*)."""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "Provider",
    "OpenAIModelName",
    "BlabladorModelName",
    "BLABLADOR_MAX_MODEL_LEN",
    "BLABLADOR_MODEL_DISPLAY_NAMES",
    "get_blablador_model_display_name",
    "PROVIDER_MODELS",
    "get_models_for_provider",
    "get_default_model_for_provider",
]


class Provider(StrEnum):
    """Stub — implemented in 01/CP1."""


class OpenAIModelName(StrEnum):
    """Stub — implemented in 01/CP1."""


class BlabladorModelName(StrEnum):
    """Stub — implemented in 01/CP1."""


BLABLADOR_MAX_MODEL_LEN: dict[str, int] = {}
BLABLADOR_MODEL_DISPLAY_NAMES: dict[str, str] = {}
PROVIDER_MODELS: dict[Provider, list[str]] = {}


def get_blablador_model_display_name(model_id: str) -> str:
    raise NotImplementedError("juena_core.schema.llm_models.get_blablador_model_display_name lands in 01/CP1")


def get_models_for_provider(provider: Provider) -> list[str]:
    raise NotImplementedError("juena_core.schema.llm_models.get_models_for_provider lands in 01/CP1")


def get_default_model_for_provider(provider: Provider) -> str:
    raise NotImplementedError("juena_core.schema.llm_models.get_default_model_for_provider lands in 01/CP1")
