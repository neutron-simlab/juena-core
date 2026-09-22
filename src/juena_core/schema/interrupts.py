"""Domain-neutral interrupt and execution-evidence schemas.

Applications add their own interrupt arms and compose their own discriminated
resume union. In particular, sandbox execution approval remains owned by
``juena-chatbot`` rather than leaking into this package.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CLARIFICATION_KIND",
    "ArtifactRef",
    "ExecutionEvidence",
    "ResumeBase",
    "ClarificationResumeInput",
]

CLARIFICATION_KIND = "clarification"


class ArtifactRef(BaseModel):
    artifact_id: str
    filename: str
    mime_type: str
    kind: Literal["image", "file"]
    size: int
    width: int | None = None
    height: int | None = None
    caption: str
    created_at: datetime
    group_id: str | None = None
    group_label: str | None = None


class ExecutionEvidence(BaseModel):
    """Facts captured before an execution tool is flattened to text."""

    model_config = ConfigDict(extra="forbid")

    graph_run_id: str = Field(min_length=1, max_length=255)
    command: str
    status: Literal[
        "completed",
        "failed",
        "timed_out",
        "busy",
        "unavailable",
        "worker_error",
        "skipped",
        "tool_error",
    ]
    exit_code: int | None = None
    truncated: bool = False
    artifact_ids: list[str] = Field(default_factory=list)
    artifact_filenames: list[str] = Field(default_factory=list)
    dropped: list[tuple[str, str]] = Field(default_factory=list)

    @property
    def attempted(self) -> bool:
        """Return whether a command actually reached an execution worker."""

        return self.status != "skipped"

    @property
    def succeeded(self) -> bool:
        return self.status == "completed" and self.exit_code == 0


class ResumeBase(BaseModel):
    """Fields shared by application-defined interrupt-resume bodies."""

    thread_id: str = Field(min_length=1, max_length=255)
    interrupt_id: str = Field(min_length=1, max_length=255)
    provider: str | None = None
    model: str | None = None


class ClarificationResumeInput(ResumeBase):
    """Answer a question an agent raised with ``ask_user``."""

    kind: Literal["clarification"]
    answer: str = Field(min_length=1, max_length=10_000)
