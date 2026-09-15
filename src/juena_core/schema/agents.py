"""Stub for 01/CP1. Ported from ``juena/schema/agents.py`` — the
verified-report contract (00-BOUNDARY.md, *Moves whole*)."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["SpecialistReport", "ResultArtifactEvidence", "AskUserSchema"]


class SpecialistReport(BaseModel):
    """Stub — implemented in 01/CP1."""


class ResultArtifactEvidence(BaseModel):
    """Stub — implemented in 01/CP1."""


class AskUserSchema(BaseModel):
    """Stub — implemented in 01/CP1."""
