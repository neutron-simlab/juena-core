"""Dependency-free attachment validation shared by UI and server.

UI checks are a courtesy to the user; the server remains the enforcing side.
Applications may supply their own extension set and limits without modifying
global core policy.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
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
    "UploadPolicy",
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
    "dat",
    "sql",
    "html",
    "css",
    "xml",
    "ipynb",
)
DEFAULT_SUFFIXES: frozenset[str] = frozenset(
    f".{suffix}" for suffix in DEFAULT_TEXT_READABLE_FILE_TYPES
)


@dataclass(frozen=True)
class UploadPolicy:
    """Application upload limits; binary files require a conversation filesystem.

    Creating a policy raises ``ValueError`` when the text and binary
    extensions overlap, a binary extension is not TIFF, an extension is not
    lowercase with a leading dot, or a limit is not positive.

    Attributes:
        text_suffixes: Extensions accepted as text, such as ``".py"``.
        binary_suffixes: Extensions accepted as binary; only ``.tif`` and
            ``.tiff``.
        max_text_bytes: Largest text file.
        max_binary_bytes: Largest binary file.
        max_total_bytes: Largest combined size of one message's files.
        max_files: Most files per message.
    """

    text_suffixes: frozenset[str] = DEFAULT_SUFFIXES
    binary_suffixes: frozenset[str] = frozenset()
    max_text_bytes: int = MAX_ATTACHMENT_BYTES
    max_binary_bytes: int = MAX_ATTACHMENT_BYTES
    max_total_bytes: int = MAX_ATTACHMENTS_PER_MESSAGE * MAX_ATTACHMENT_BYTES
    max_files: int = MAX_ATTACHMENTS_PER_MESSAGE

    def __post_init__(self) -> None:
        if self.text_suffixes & self.binary_suffixes:
            raise ValueError("Text and binary upload extensions must be disjoint")
        if self.binary_suffixes - {".tif", ".tiff"}:
            raise ValueError("Only TIFF binary uploads are supported")
        if any(s != s.lower() or not s.startswith(".") for s in self.suffixes):
            raise ValueError("Upload extensions must be lowercase and start with a dot")
        if min(self.max_text_bytes, self.max_binary_bytes, self.max_total_bytes, self.max_files) <= 0:
            raise ValueError("Upload limits must be positive")

    @property
    def suffixes(self) -> frozenset[str]:
        """Every accepted extension, text and binary."""
        return self.text_suffixes | self.binary_suffixes

    def file_limit(self, filename: str) -> int:
        """Return the size limit for one file.

        Args:
            filename: The file's name; its extension picks the limit.

        Returns:
            ``max_binary_bytes`` for a binary extension, else ``max_text_bytes``.
        """
        return self.max_binary_bytes if Path(filename).suffix.lower() in self.binary_suffixes else self.max_text_bytes


def is_text_readable_filename(
    filename: str | None,
    *,
    allowed_suffixes: Collection[str] = DEFAULT_SUFFIXES,
) -> bool:
    """Return whether *filename* uses one of the supplied extensions.

    Args:
        filename: The file's name; ``None`` counts as no extension.
        allowed_suffixes: Accepted extensions, lowercase with a leading dot.

    Returns:
        Whether the lowercased extension is accepted.
    """

    return Path(filename or "").suffix.lower() in allowed_suffixes


def validate_attachments(
    attachments: Sequence[tuple[str, int | None]],
    *,
    allowed_suffixes: Collection[str] = DEFAULT_SUFFIXES,
    max_bytes: int = MAX_ATTACHMENT_BYTES,
    max_files: int = MAX_ATTACHMENTS_PER_MESSAGE,
    upload_policy: UploadPolicy | None = None,
) -> list[str]:
    """Return human-readable reasons why *attachments* would be rejected.

    Args:
        attachments: ``(filename, size)`` pairs; a ``None`` size is not checked.
        allowed_suffixes: Accepted extensions.
        max_bytes: Largest file.
        max_files: Most files.
        upload_policy: When given, its extensions and limits replace the three
            arguments above, and the combined size is checked too.

    Returns:
        One message per problem; empty when the attachments are acceptable.
    """

    errors: list[str] = []
    if upload_policy is not None:
        allowed_suffixes = upload_policy.suffixes
        max_files = upload_policy.max_files

    if len(attachments) > max_files:
        errors.append(
            f"At most {max_files} files may be uploaded per message "
            f"({len(attachments)} selected)."
        )

    for filename, size in attachments:
        file_limit = upload_policy.file_limit(filename) if upload_policy else max_bytes
        if not is_text_readable_filename(
            filename,
            allowed_suffixes=allowed_suffixes,
        ):
            errors.append(
                f"'{filename}' has an unsupported extension. Only configured "
                "file types are allowed."
            )
        if size is not None and size > file_limit:
            max_megabytes = file_limit / 1024 / 1024
            errors.append(
                f"'{filename}' is {size / 1024 / 1024:.1f} MB, over the "
                f"{max_megabytes:g} MB per-file limit."
            )

    if upload_policy and sum(size or 0 for _, size in attachments) > upload_policy.max_total_bytes:
        errors.append(f"Attachments exceed the {upload_policy.max_total_bytes / 1024**2:g} MiB combined limit.")
    return errors
