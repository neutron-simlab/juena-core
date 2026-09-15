"""juena_core — shared agent/server/UI infrastructure extracted from juena-chatbot.

See ``00-BOUNDARY.md`` for the disposition table and the boundary decisions
this package's shape follows from. ``[ui]`` (``juena_core.ui``), ``[mcp]``
(``juena_core.mcp``), and ``[sandbox]`` (``juena_core.sandbox``) are optional
extras and are deliberately not imported here.
"""

from __future__ import annotations

from juena_core.config import CoreSettings, configure, settings

__version__ = "0.1.0"

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
]
