"""Dependency-free sandbox statuses and per-execution artifact limits."""

from __future__ import annotations

from typing import Literal

from juena_core.artifacts import (
    DOWNLOAD_TYPES,
    MAX_ARTIFACT_BYTES,
    MAX_FILE_ARTIFACT_BYTES,
)

__all__ = [
    "JobStatus",
    "ACTIVE_STATUSES",
    "TERMINAL_STATUSES",
    "MAX_ARTIFACTS_PER_EXECUTION",
    "MAX_FILE_ARTIFACTS_PER_EXECUTION",
    "COLLECTED_EXTENSIONS",
    "MAX_ARTIFACT_BYTES",
    "MAX_FILE_ARTIFACT_BYTES",
]

JobStatus = Literal["queued", "running", "completed", "failed", "timed_out"]

ACTIVE_STATUSES = ("queued", "running")
TERMINAL_STATUSES = ("completed", "failed", "timed_out")

# The worker selects a bounded set of filenames from its workspace. The
# application validates the bytes again through ArtifactStore before delivery.
MAX_ARTIFACTS_PER_EXECUTION = 4
MAX_FILE_ARTIFACTS_PER_EXECUTION = 8
COLLECTED_EXTENSIONS = frozenset(DOWNLOAD_TYPES) | {".png"}
