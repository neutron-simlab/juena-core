"""Optional rootless-Podman execution for applications built on juena-core.

Nothing in this package is imported by :mod:`juena_core` itself. Applications
that need execution install ``juena-core[sandbox]``, configure the runtime,
and wire its backend, interrupt, workspace, and lifespan explicitly.
"""

from __future__ import annotations

__all__ = [
    "approvals",
    "backend",
    "config",
    "constants",
    "evidence",
    "executor",
    "jobs",
    "middleware",
    "models",
    "policy",
    "runtime",
    "worker",
    "workspace",
]
