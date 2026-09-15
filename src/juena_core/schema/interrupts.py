"""Stub for 01/CP1. The generic half of ``juena/schema/sandbox.py``
(00-BOUNDARY.md, decision 5).

``EXECUTE_APPROVAL_KIND`` and ``ApprovalResumeInput`` stay in juena-chatbot's
own sandbox schema — they are Podman-shaped. ``SandboxExecutionEvidence``
becomes ``ExecutionEvidence`` here, fields unchanged.

**The ``ResumeInput`` union is not defined here.** An earlier revision moved
the union into core while leaving one of its arms in the application, which
is impossible — the union is
``Annotated[ApprovalResumeInput | ClarificationResumeInput, Field(discriminator="kind")]``.
Core exports the base and the clarification arm; each application composes
its own discriminated union from the arms it actually has. v2 has no
execute-approval interrupt, so its union is just ``ClarificationResumeInput``.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["CLARIFICATION_KIND", "ArtifactRef", "ExecutionEvidence", "ResumeBase", "ClarificationResumeInput"]

CLARIFICATION_KIND = "clarification"


class ArtifactRef(BaseModel):
    """Stub — implemented in 01/CP1."""


class ExecutionEvidence(BaseModel):
    """Stub — implemented in 01/CP1. Renamed from ``SandboxExecutionEvidence``;
    fields (``attempted``, ``succeeded``, ``status``, ``exit_code``) unchanged."""


class ResumeBase(BaseModel):
    """Stub — implemented in 01/CP1. Renamed from ``_ResumeBase`` (no longer
    private: applications compose their own union from this base)."""


class ClarificationResumeInput(ResumeBase):
    """Stub — implemented in 01/CP1."""
