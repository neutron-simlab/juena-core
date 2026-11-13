"""Helper functions for staged chat input parsing and manifests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from deepagents.backends.utils import file_data_to_string

from juena_core.server.chat.input_constants import (
    ERROR_LINE_RE,
    FENCED_CODE_RE,
    LANGUAGE_SUFFIXES,
    UPLOADS_MANIFEST_PATH,
    UPLOADS_PREFIX,
)
from juena_core.server.chat.input_types import PastedCodeContext, UploadedAttachment

__all__ = [
    "extract_pasted_code_context",
    "staged_code_path",
    "is_persistent_input_path",
    "build_uploads_manifest",
    "build_inputs_manifest",
]


def _line_is_error_like(line: str) -> bool:
    return bool(ERROR_LINE_RE.search(line))


def _block_is_error(body: str) -> bool:
    """Whether a fenced block reads as a traceback rather than source code.

    A majority vote over non-blank lines, because a traceback interleaves error
    lines with echoed source lines ("    x = 1") that match nothing, while real
    source rarely has any error-shaped line at all.
    """

    lines = [line for line in body.splitlines() if line.strip()]
    if not lines:
        return False
    hits = sum(1 for line in lines if _line_is_error_like(line))
    return hits * 2 >= len(lines)


def _compact_summary(text: str, *, max_chars: int = 500) -> str:
    compact = " ".join(text.split())
    if not compact:
        return "(no typed prompt provided)"
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def extract_pasted_code_context(message: str) -> PastedCodeContext | None:
    """Extract code and errors from fenced blocks in typed chat text.

    Only ``` fenced blocks count. Guessing at unfenced text misreads ordinary
    prose as code ("What does the Error: message in the manual mean?" looks like
    a traceback), which silently reframes the whole turn. A fence is something
    the user types on purpose.

    Unfenced text is not staged, but the message still reaches the agent
    verbatim.

    Args:
        message: The text the user typed.

    Returns:
        The extracted code, errors and remaining goal text, or None when the
        message has no non-empty fenced block.
    """

    if not message.strip():
        return None

    fenced_blocks = [
        block for block in FENCED_CODE_RE.finditer(message) if block.group("body").strip()
    ]
    if not fenced_blocks:
        return None

    code_bodies: list[str] = []
    error_bodies: list[str] = []
    for block in fenced_blocks:
        body = block.group("body").strip("\n")
        (error_bodies if _block_is_error(body) else code_bodies).append(body)

    # The first fence with a language tag sets the staged code file's suffix
    # (see `staged_code_path`).
    language = next(
        (
            block.group("lang").strip().lower()
            for block in fenced_blocks
            if block.group("lang").strip()
        ),
        None,
    )
    remaining = FENCED_CODE_RE.sub("", message)
    goal = "\n".join(line for line in remaining.splitlines() if line.strip()).strip()

    return PastedCodeContext(
        user_goal=goal,
        code="\n\n".join(code_bodies).strip(),
        error_text="\n\n".join(error_bodies).strip(),
        language=language,
    )


def staged_code_path(context: PastedCodeContext) -> str:
    """Return the canonical staged code path for extracted inline code.

    Args:
        context: The pasted code; its language picks the file suffix.

    Returns:
        ``/inputs/current_code`` plus the language's suffix, or ``.txt`` when
        the language is unknown.
    """

    suffix = ".txt"
    if context.language:
        suffix = LANGUAGE_SUFFIXES.get(context.language.lower(), ".txt")
    return f"/inputs/current_code{suffix}"


def is_persistent_input_path(path: str) -> bool:
    """Whether a staged path is kept from one turn to the next.

    Args:
        path: A path under ``/inputs``.

    Returns:
        True for uploads and the uploads manifest; other staged files last
        one turn.
    """

    return path.startswith(UPLOADS_PREFIX) or path == UPLOADS_MANIFEST_PATH


def _format_upload_manifest_entry(
    index: int,
    *,
    staged_path: str,
    original_filename: str,
    char_count: int | None,
    size_bytes: int | None = None,
) -> str:
    size = f"{char_count} chars" if char_count is not None else f"{size_bytes} bytes"
    return f"{index}. `{staged_path}` (from {original_filename}, {size})"


def build_uploads_manifest(
    existing_files: dict[str, Any],
    existing_upload_paths: Sequence[str],
    new_uploads: Sequence[UploadedAttachment],
) -> tuple[str, str | None] | None:
    """Create or update the persistent thread upload manifest.

    Args:
        existing_files: The thread's current ``/inputs`` files.
        existing_upload_paths: Upload paths staged in earlier turns.
        new_uploads: Uploads accepted this turn.

    Returns:
        The manifest text and the manifest's original creation time, or None
        when there is nothing to list or add.
    """

    existing_manifest = existing_files.get(UPLOADS_MANIFEST_PATH)
    created_at = None
    if isinstance(existing_manifest, dict):
        created_at = existing_manifest.get("created_at")
        existing_manifest_text = file_data_to_string(existing_manifest).rstrip()
    else:
        existing_manifest_text = ""

    if existing_manifest_text:
        if not new_uploads:
            return None
        start_index = len(existing_upload_paths) + 1
        new_entries = [
            _format_upload_manifest_entry(
                start_index + index,
                staged_path=upload.staged_path,
                original_filename=upload.original_filename,
                char_count=upload.char_count,
                size_bytes=upload.size_bytes,
            )
            for index, upload in enumerate(new_uploads)
        ]
        return "\n".join([existing_manifest_text, *new_entries]), created_at

    upload_by_path = {upload.staged_path: upload for upload in new_uploads}
    manifest_paths = sorted(existing_upload_paths)
    manifest_paths.extend(
        upload.staged_path
        for upload in new_uploads
        if upload.staged_path not in existing_upload_paths
    )
    if not manifest_paths:
        return None

    entries: list[str] = []
    for index, path in enumerate(manifest_paths, start=1):
        upload = upload_by_path.get(path)
        if upload is not None:
            entries.append(
                _format_upload_manifest_entry(
                    index,
                    staged_path=upload.staged_path,
                    original_filename=upload.original_filename,
                    char_count=upload.char_count,
                    size_bytes=upload.size_bytes,
                )
            )
            continue

        file_data = existing_files.get(path)
        char_count = len(file_data_to_string(file_data)) if isinstance(file_data, dict) else 0
        entries.append(
            _format_upload_manifest_entry(
                index,
                staged_path=path,
                original_filename=Path(path).name,
                char_count=char_count,
            )
        )

    lines = [
        "# Persistent chat uploads",
        "",
        "Files uploaded in this chat remain available until the chat is deleted.",
        "",
        *entries,
    ]
    return "\n".join(lines), created_at


def build_inputs_manifest(
    raw_message: str,
    attachments: Sequence[UploadedAttachment],
    pasted_context: PastedCodeContext | None,
    *,
    has_thread_uploads: bool,
    closing_note: str,
    uploads_prefix: str = UPLOADS_PREFIX,
) -> str:
    """Build the short staged-input manifest injected into agent history.

    Args:
        raw_message: The text the user typed.
        attachments: Uploads accepted this turn.
        pasted_context: Code and errors pasted in fences, if any.
        has_thread_uploads: Whether the thread has persistent uploads.
        closing_note: The last paragraph. It comes from the application,
            because what an agent can do with the staged paths differs per
            application.
        uploads_prefix: Where uploads live; replaces ``/inputs/uploads/``
            throughout the text.

    Returns:
        The manifest the model reads in place of the typed message.
    """

    summary = pasted_context.user_goal if pasted_context and pasted_context.user_goal else raw_message
    lines: list[str] = []

    if has_thread_uploads:
        lines.extend(
            [
                "Persistent uploaded files for this chat are available under `/inputs/uploads/`.",
                f"Inspect `{UPLOADS_MANIFEST_PATH}` to see every upload in this thread.",
            ]
        )

    if attachments or pasted_context is not None:
        lines.extend(
            [
                "This message also staged turn-scoped helper files under `/inputs`.",
                "Inspect `/inputs` before answering.",
            ]
        )
    elif has_thread_uploads:
        lines.append(
            "Inspect `/inputs/uploads/` before answering if the user refers to uploaded material."
        )

    lines.extend(["", f"User request: {_compact_summary(summary)}"])

    if has_thread_uploads:
        lines.extend(["", "Persistent files:", f"- {UPLOADS_MANIFEST_PATH}"])

    if attachments or pasted_context is not None:
        lines.extend(["", "Current turn files:", "- /inputs/current_message.txt"])

    if pasted_context and pasted_context.contains_code:
        lines.append(f"- {staged_code_path(pasted_context)}")
    if pasted_context and pasted_context.contains_error:
        lines.append("- /inputs/current_error.txt")

    for attachment in attachments:
        lines.append(f"- {attachment.staged_path} ({attachment.description})")

    lines.extend(["", closing_note])
    return "\n".join(lines).replace("/inputs/uploads/", uploads_prefix)
