"""Typed containers for staged chat input handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["UploadedAttachment", "PastedCodeContext", "PreparedCodeChatInputs"]


@dataclass(frozen=True)
class UploadedAttachment:
    """Validated attachment metadata staged in agent state.

    Attributes:
        original_filename: The file name the user uploaded.
        staged_path: Where the file is staged.
        char_count: Characters in a text upload; None for a binary one.
        size_bytes: Size in bytes.
        media_type: The upload's media type, such as ``text/plain``.
    """

    original_filename: str
    staged_path: str
    char_count: int | None = None
    size_bytes: int | None = None
    media_type: str | None = None

    @property
    def description(self) -> str:
        """The manifest's note on this file, such as ``from data.txt, 120 chars``."""

        size = f"{self.char_count} chars" if self.char_count is not None else f"{self.size_bytes} bytes"
        return f"from {self.original_filename}, {size}"

    def display_metadata(self) -> dict[str, Any]:
        """Describe the upload for the transcript.

        Returns:
            The file name plus whichever of character count, size and media
            type are known.
        """

        metadata: dict[str, Any] = {"name": self.original_filename}
        if self.char_count is not None:
            metadata["chars"] = self.char_count
        if self.size_bytes is not None:
            metadata["size_bytes"] = self.size_bytes
        if self.media_type is not None:
            metadata["media_type"] = self.media_type
        return metadata


@dataclass(frozen=True)
class PastedCodeContext:
    """Normalized code/error content extracted from the typed prompt.

    Attributes:
        user_goal: The typed text outside the fenced blocks.
        code: The fenced blocks that read as source code, joined.
        error_text: The fenced blocks that read as tracebacks, joined.
        language: The first language tag on a fence, if any.
    """

    user_goal: str
    code: str
    error_text: str
    language: str | None = None

    @property
    def contains_code(self) -> bool:
        """Whether any code was pasted."""

        return bool(self.code.strip())

    @property
    def contains_error(self) -> bool:
        """Whether any error text was pasted."""

        return bool(self.error_text.strip())


@dataclass(frozen=True)
class PreparedCodeChatInputs:
    """Message override plus staged file updates for one chat turn.

    Attributes:
        message_override: What the model reads in place of the typed text,
            usually the staged-input manifest.
        files_update: The update for the graph's ``files`` channel; a None
            value deletes that path.
        attachments: Uploads accepted this turn, recorded on the human message
            so the transcript shows which files were attached, also after a
            reload.
        workspace_files: The fully merged ``/inputs`` view, for an application
            that materialises these files outside graph state. Unlike
            ``files_update``, it includes persistent files from earlier turns
            and leaves out deletion markers. Core only computes it; the
            application's ``ThreadWorkspace`` (see
            ``juena_core.server.api.endpoints``) decides what happens to it.
    """

    message_override: str
    files_update: dict[str, Any | None]
    attachments: tuple[UploadedAttachment, ...] = ()
    workspace_files: dict[str, Any] | None = None
