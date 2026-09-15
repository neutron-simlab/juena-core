"""Stub for 01/CP1. Ported from ``juena/schema/upload_limits.py``, keeping
its "deliberately dependency-free" property.

**One change** (00-BOUNDARY.md, decision 7): ``TEXT_READABLE_FILE_TYPES``
becomes a default argument rather than a module constant, because v2 must
accept ``.dat``, ``.inf``, ``.nxs`` and ``.h5``, which are meaningless to
juena:

    validate_attachments(attachments, *, allowed_suffixes=DEFAULT_SUFFIXES, ...)
"""

from __future__ import annotations

__all__ = [
    "CHAT_INPUT_MAX_CHARS",
    "CHAT_INPUT_MAX_UPLOAD_MB",
    "MAX_ATTACHMENTS_PER_MESSAGE",
    "MAX_ATTACHMENT_BYTES",
    "TEXT_READABLE_FILE_TYPES",
    "ALLOWED_SUFFIXES",
    "is_text_readable_filename",
    "validate_attachments",
]

CHAT_INPUT_MAX_CHARS = 4_000
CHAT_INPUT_MAX_UPLOAD_MB = 1
MAX_ATTACHMENTS_PER_MESSAGE = 5
MAX_ATTACHMENT_BYTES = CHAT_INPUT_MAX_UPLOAD_MB * 1024 * 1024

TEXT_READABLE_FILE_TYPES: list[str] = []
ALLOWED_SUFFIXES: set[str] = set()


def is_text_readable_filename(filename: str | None) -> bool:
    raise NotImplementedError("juena_core.schema.upload_limits.is_text_readable_filename lands in 01/CP1")


def validate_attachments(attachments, *, allowed_suffixes=ALLOWED_SUFFIXES, **kwargs):
    raise NotImplementedError("juena_core.schema.upload_limits.validate_attachments lands in 01/CP1")
