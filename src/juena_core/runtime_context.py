"""Helpers for reading typed application runtime context."""

from __future__ import annotations

from typing import Any

__all__ = ["_context_value"]


def _context_value(context: Any, name: str) -> str | None:
    """Read a non-empty context value from a mapping or object."""

    if isinstance(context, dict):
        value = context.get(name)
    else:
        value = getattr(context, name, None)
    return str(value) if value else None
