"""Environment-owned configuration for the simple chatbot example.

The example is an application, so it reads the environment and translates it
to ``CoreSettings``. The reusable ``juena_core`` package intentionally does
neither of those things.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from uuid import UUID

from juena_core.config import CoreSettings
from juena_core.schema.llm_models import Provider, get_default_model_for_provider

DEMO_USER_ID = UUID("a44d07f8-4e25-4cf9-8b80-f86bbcc65d68")


def build_settings(environ: Mapping[str, str] | None = None) -> CoreSettings:
    """Build the demo's core settings from *environ*.

    Binding and publication are intentionally not configurable: this example
    uses one fixed local identity and must remain reachable only on loopback.
    Replace the principal in ``service.py`` before publishing a derived app.
    """

    env = os.environ if environ is None else environ
    provider_name = env.get("DEMO_PROVIDER", Provider.OPENAI.value).lower()
    try:
        provider = Provider(provider_name)
    except ValueError as exc:
        choices = ", ".join(item.value for item in Provider)
        raise RuntimeError(f"DEMO_PROVIDER must be one of: {choices}") from exc

    model = env.get("DEMO_MODEL") or get_default_model_for_provider(provider)
    data_dir = Path(env.get("DEMO_DATA_DIR", ".demo-data")).expanduser().resolve()

    return CoreSettings(
        OPENAI_API_KEY=env.get("OPENAI_API_KEY"),
        BLABLADOR_API_KEY=env.get("BLABLADOR_API_KEY"),
        BLABLADOR_BASE_URL=env.get("BLABLADOR_BASE_URL"),
        MAX_TOKENS=4_096,
        TIMEOUT_SECONDS=120,
        MAX_RETRIES=2,
        DEFAULT_PROVIDER=provider.value,
        DEFAULT_MODEL=model,
        OPENAI_AVAILABLE_MODELS=None,
        BLABLADOR_AVAILABLE_MODELS=None,
        OPENAI_DEFAULT_MODEL=get_default_model_for_provider(Provider.OPENAI),
        BLABLADOR_DEFAULT_MODEL=get_default_model_for_provider(Provider.BLABLADOR),
        STREAM_TOOL_PAYLOADS=False,
        DATABASE_URL=env.get(
            "DATABASE_URL",
            "postgresql://juena_demo:juena_demo@127.0.0.1:55433/juena_demo",
        ),
        DATABASE_POOL_MAX_SIZE=5,
        SESSION_TTL_HOURS=24,
        SESSION_COOKIE_SECURE=False,
        FALLBACK_PROVIDER=None,
        EXECUTE_TIMEOUT_SECONDS=300,
        ARTIFACT_ROOT=data_dir / "artifacts",
        AUDIT_FILE=data_dir / "audit.jsonl",
        LOG_LEVEL=env.get("LOG_LEVEL", "INFO"),
        LOG_DIR=data_dir / "logs",
        BIND_HOST="127.0.0.1",
        API_PUBLISHED=False,
    )
