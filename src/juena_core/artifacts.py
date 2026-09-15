"""Stub for 01/CP2. Lifted from ``juena/sandbox/artifacts.py``
(00-BOUNDARY.md, decision 5) — a per-user file store with per-turn budgets,
PNG validation and an audit log. This is *delivery*, not sandboxing, which is
why v2 needs it for monitor1D/2D plots. Its only sandbox import is six
constants from ``sandbox/constants.py``, which come with it.
"""

from __future__ import annotations

__all__ = [
    "ARTIFACT_MESSAGE_KEY",
    "MAX_IMAGE_EDGE",
    "MAX_IMAGE_PIXELS",
    "MAX_UNDELIVERED_NOTICES_PER_TURN",
    "ArtifactStore",
    "get_artifact_store",
    "set_artifact_store_for_tests",
    "undelivered_note",
]

MAX_IMAGE_EDGE = 4096
MAX_IMAGE_PIXELS = 16_000_000
ARTIFACT_MESSAGE_KEY = "juena_artifacts"
MAX_UNDELIVERED_NOTICES_PER_TURN = 8


class _StoredArtifact:
    """Stub — implemented in 01/CP2."""


class _PendingTurn:
    """Stub — implemented in 01/CP2."""


class ArtifactStore:
    """Stub — implemented in 01/CP2."""


def undelivered_note(dropped: list[tuple[str, str]]) -> str:
    raise NotImplementedError("juena_core.artifacts.undelivered_note lands in 01/CP2")


def get_artifact_store() -> ArtifactStore:
    raise NotImplementedError("juena_core.artifacts.get_artifact_store lands in 01/CP2")


def set_artifact_store_for_tests(store: ArtifactStore | None) -> None:
    raise NotImplementedError("juena_core.artifacts.set_artifact_store_for_tests lands in 01/CP2")
