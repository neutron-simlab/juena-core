"""juena_core — shared agent, server and UI building blocks for applications.

``[ui]`` (``juena_core.ui``), ``[mcp]`` (``juena_core.mcp``), and ``[sandbox]``
(``juena_core.sandbox``) are optional extras and are deliberately not imported
here.
"""

from __future__ import annotations

from juena_core.config import CoreSettings, configure, settings

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "CoreSettings",
    "configure",
    "settings",
    "config",
    "log",
    "llms_providers",
    "runtime_context",
    "artifacts",
    "schema",
    "agents",
    "clients",
    "server",
    "sandbox",
    "tracing",
]
