"""Stub for 01/CP2. Ported from ``juena/agents/findings.py`` — the
``/findings/`` contract. Its docstring names four callers, two of which stay
in the application; update the docstring when this is filled in
(00-BOUNDARY.md, *Moves whole*). Imports ``FINDINGS_PREFIX`` from
``.backends``, which moves alongside it.
"""

from __future__ import annotations

from typing import Any, Mapping

from juena_core.agents.backends import FINDINGS_PREFIX

__all__ = [
    "CROSSING_PREFIXES",
    "CONFLICT_PREFIX",
    "is_conflict_path",
    "conflict_path",
    "crossing_files",
    "findings_delta",
    "fingerprint",
    "validate_findings",
]

CROSSING_PREFIXES: tuple[str, ...] = ("/inputs/", FINDINGS_PREFIX)
CONFLICT_PREFIX = f"{FINDINGS_PREFIX}_conflicts/"


def is_conflict_path(path: str) -> bool:
    raise NotImplementedError("juena_core.agents.findings.is_conflict_path lands in 01/CP2")


def conflict_path(path: str, job_id: Any) -> str:
    raise NotImplementedError("juena_core.agents.findings.conflict_path lands in 01/CP2")


def crossing_files(files: Mapping[str, Any] | None) -> dict[str, Any]:
    raise NotImplementedError("juena_core.agents.findings.crossing_files lands in 01/CP2")


def findings_delta(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.agents.findings.findings_delta lands in 01/CP2")


def fingerprint(files: Mapping[str, Any] | None) -> dict[str, str]:
    raise NotImplementedError("juena_core.agents.findings.fingerprint lands in 01/CP2")


def validate_findings(value: Any) -> Any:
    raise NotImplementedError("juena_core.agents.findings.validate_findings lands in 01/CP2")
