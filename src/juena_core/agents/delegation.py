"""Stub for 01/CP2. Ported from ``juena/agents/delegation.py`` verbatim
(00-BOUNDARY.md, *Moves whole*).

Carries only ``messages`` and ``files`` across the specialist boundary in
both directions. v2's typed ``module_results`` channel is deliberately not
here — 00-BOUNDARY.md, decision 14: that channel is application-specific and
lives in ``vitess_ai``, not core.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable

__all__ = ["SpecialistDelegate", "with_delegation_boundary"]


class SpecialistDelegate(Runnable[dict[str, Any], dict[str, Any]]):
    """Stub — implemented in 01/CP2."""


def with_delegation_boundary(specialists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raise NotImplementedError("juena_core.agents.delegation.with_delegation_boundary lands in 01/CP2")
