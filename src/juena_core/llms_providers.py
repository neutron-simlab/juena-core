"""Stub for 01/CP1. Ported from ``juena/core/llms_providers.py``.

``get_config()`` becomes ``config.settings()`` throughout — see
00-BOUNDARY.md, decision 1.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "get_available_providers",
    "build_chat_model",
    "get_available_models",
    "get_default_model",
    "format_model_name",
]


def _is_available(provider: str) -> bool:
    raise NotImplementedError("juena_core.llms_providers._is_available lands in 01/CP1")


def get_available_providers() -> dict[str, bool]:
    raise NotImplementedError("juena_core.llms_providers.get_available_providers lands in 01/CP1")


def _model_kwargs(provider: str, model: str, temperature: float, overrides: dict) -> dict:
    raise NotImplementedError("juena_core.llms_providers._model_kwargs lands in 01/CP1")


def _build(provider: str, model: str, temperature: float, overrides: dict) -> Any:
    raise NotImplementedError("juena_core.llms_providers._build lands in 01/CP1")


def _build_cached(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.llms_providers._build_cached lands in 01/CP1")


def build_chat_model(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.llms_providers.build_chat_model lands in 01/CP1")


def get_available_models(provider: str) -> list[str]:
    raise NotImplementedError("juena_core.llms_providers.get_available_models lands in 01/CP1")


def get_default_model(provider: str) -> str:
    raise NotImplementedError("juena_core.llms_providers.get_default_model lands in 01/CP1")


def format_model_name(provider: str, model: str) -> str:
    raise NotImplementedError("juena_core.llms_providers.format_model_name lands in 01/CP1")
