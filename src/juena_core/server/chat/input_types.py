"""Typed containers for staged chat input handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["UploadedAttachment", "PastedCodeContext", "PreparedCodeChatInputs"]


@dataclass(frozen=True)
class UploadedAttachment:
    """Validated attachment metadata staged in agent state."""

    original_filename: str
    staged_path: str
    char_count: int


@dataclass(frozen=True)
class PastedCodeContext:
    """Normalized code/error content extracted from the typed prompt."""

    user_goal: str
    code: str
    error_text: str
    language: str | None = None

    @property
    def contains_code(self) -> bool:
        return bool(self.code.strip())

    @property
    def contains_error(self) -> bool:
        return bool(self.error_text.strip())


@dataclass(frozen=True)
class PreparedCodeChatInputs:
    """Message override plus staged file updates for one chat turn."""

    message_override: str
    files_update: dict[str, Any | None]
    # Uploads accepted this turn, recorded on the human message so the transcript
    # can show which files were attached -- including after a reload.
    attachments: tuple[UploadedAttachment, ...] = ()
    # Fully merged `/inputs` view for an application that has to materialise
    # these files outside graph state. Unlike `files_update`, this includes
    # persistent files from earlier turns and excludes deletion markers. Core
    # only computes it; what happens to it is the application's
    # `ThreadWorkspace` (see ``juena_core.server.api.endpoints``).
    workspace_files: dict[str, Any] | None = None
