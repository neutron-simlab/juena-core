"""Core's configuration seam. Stub for 01/CP1. See 00-BOUNDARY.md, decision 1.

New in core — there is no equivalent module in ``juena/core/config.py`` to
port. ``CoreSettings`` reads nothing itself; the application reads its own
environment and calls ``configure()`` once at startup. ``settings()`` raises
until that happens — the same shape ``get_checkpointer()`` and
``get_pool()`` already use, so a logger or a database call fails loudly at
startup rather than silently defaulting.

No ``os.getenv``. No ``load_dotenv``. No validation that prints.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CoreSettings", "configure", "settings"]


@dataclass(frozen=True, slots=True)
class CoreSettings:
    """The ~22 fields the modules that move actually read. Filled in 01/CP1
    from the table in 00-BOUNDARY.md, decision 1."""


_settings: CoreSettings | None = None


def configure(settings_: CoreSettings) -> None:
    raise NotImplementedError("juena_core.config.configure lands in 01/CP1")


def settings() -> CoreSettings:
    raise NotImplementedError("juena_core.config.settings lands in 01/CP1")
