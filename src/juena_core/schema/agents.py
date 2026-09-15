"""Schemas for what agents return and what they ask for."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["SpecialistReport", "ResultArtifactEvidence", "AskUserSchema"]


class SpecialistReport(BaseModel):
    """The model-authored research package handed to the supervisor.

    Artifact metadata is deliberately absent: the server appends verified
    artifacts so a specialist cannot invent a file it did not produce.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "blocked", "failed"]
    finding: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("finding")
    @classmethod
    def strip_finding(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("finding must not be blank")
        return cleaned


class ResultArtifactEvidence(BaseModel):
    """Authenticated result-artifact metadata read from the artifact store."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    filename: str
    kind: Literal["image", "file"]
    mime_type: str
    caption: str


class AskUserSchema(BaseModel):
    """Input schema for the ``ask_user`` tool, shown to the model."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(
        min_length=1,
        description=(
            "One specific question, self-contained enough to answer without "
            "scrollback."
        ),
    )
    options: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 4 concrete suggested answers. Omit when the answer is open-ended."
        ),
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned
