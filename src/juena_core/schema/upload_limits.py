"""Dependency-free attachment validation shared by UI and server.

UI checks are a courtesy to the user; the server remains the enforcing side.
Applications may supply their own extension set and limits without modifying
global core policy.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

__all__ = [
    "CHAT_INPUT_MAX_CHARS",
    "CHAT_INPUT_MAX_UPLOAD_MB",
    "MAX_ATTACHMENTS_PER_MESSAGE",
    "MAX_ATTACHMENT_BYTES",
    "DEFAULT_TEXT_READABLE_FILE_TYPES",
    "DEFAULT_SUFFIXES",
    "is_text_readable_filename",
    "validate_attachments",
]

CHAT_INPUT_MAX_CHARS = 4_000
CHAT_INPUT_MAX_UPLOAD_MB = 1
MAX_ATTACHMENTS_PER_MESSAGE = 5
MAX_ATTACHMENT_BYTES = CHAT_INPUT_MAX_UPLOAD_MB * 1024 * 1024

DEFAULT_TEXT_READABLE_FILE_TYPES: tuple[str, ...] = (
    "py",
    "js",
    "ts",
    "java",
    "c",
    "cpp",
    "rs",
    "go",
    "sh",
    "md",
    "txt",
    "log",
    "json",
    "yaml",
    "yml",
    "toml",
    "ini",
    "cfg",
    "csv",
    "sql",
    "html",
    "css",
    "xml",
    "ipynb",
)
DEFAULT_SUFFIXES: frozenset[str] = frozenset(
    f".{suffix}" for suffix in DEFAULT_TEXT_READABLE_FILE_TYPES
)


def is_text_readable_filename(
    filename: str | None,
    *,
    allowed_suffixes: Collection[str] = DEFAULT_SUFFIXES,
) -> bool:
    """Return whether *filename* uses one of the supplied extensions."""

    return Path(filename or "").suffix.lower() in allowed_suffixes


def validate_attachments(
    attachments: Sequence[tuple[str, int | None]],
    *,
    allowed_suffixes: Collection[str] = DEFAULT_SUFFIXES,
    max_bytes: int = MAX_ATTACHMENT_BYTES,
    max_files: int = MAX_ATTACHMENTS_PER_MESSAGE,
) -> list[str]:
    """Return human-readable reasons why *attachments* would be rejected."""

    errors: list[str] = []

    if len(attachments) > max_files:
        errors.append(
            f"At most {max_files} files may be uploaded per message "
            f"({len(attachments)} selected)."
        )

    for filename, size in attachments:
        if not is_text_readable_filename(
            filename,
            allowed_suffixes=allowed_suffixes,
        ):
            errors.append(
                f"'{filename}' has an unsupported extension. Only configured "
                "file types are allowed."
            )
        if size is not None and size > max_bytes:
            max_megabytes = max_bytes / 1024 / 1024
            errors.append(
                f"'{filename}' is {size / 1024 / 1024:.1f} MB, over the "
                f"{max_megabytes:g} MB per-file limit."
            )

    return errors
