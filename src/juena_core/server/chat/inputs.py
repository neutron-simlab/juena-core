"""Staging typed snippets and uploaded attachments into agent state.

One turn's uploads become files under ``/inputs/uploads/`` in the graph's file
state, plus a manifest the model is told to read. Nothing here knows how those
files reach a shell or a simulation binary: that is the application's
``ThreadWorkspace``, and it reads :attr:`PreparedCodeChatInputs.workspace_files`.
"""

from __future__ import annotations

import re
from dataclasses import replace
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

from deepagents.backends.utils import create_file_data
from fastapi import UploadFile
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from juena_core.filesystem.service import ThreadFiles
from juena_core.log import get_logger
from juena_core.schema.upload_limits import (
    MAX_ATTACHMENT_BYTES,
    UploadPolicy,
    validate_attachments,
)
from juena_core.server.chat.input_constants import (
    INPUTS_PREFIX as _INPUTS_PREFIX,
)
from juena_core.server.chat.input_constants import (
    UPLOADS_MANIFEST_PATH as _UPLOADS_MANIFEST_PATH,
)
from juena_core.server.chat.input_constants import (
    UPLOADS_PREFIX as _UPLOADS_PREFIX,
)
from juena_core.server.chat.input_types import PreparedCodeChatInputs, UploadedAttachment
from juena_core.server.chat.input_utils import (
    build_inputs_manifest,
    build_uploads_manifest,
    extract_pasted_code_context,
)
from juena_core.server.chat.input_utils import (
    is_persistent_input_path as _is_persistent_input_path,
)
from juena_core.server.chat.input_utils import (
    staged_code_path,
)

logger = get_logger(__name__)

__all__ = [
    "sanitize_uploaded_filename",
    "normalize_uploaded_attachments",
    "get_existing_input_files",
    "prepare_code_chat_turn_inputs",
]

# Read in chunks so an oversized upload is refused after one chunk rather than
# after the whole body has been buffered into memory.
_UPLOAD_CHUNK_BYTES = 64 * 1024


def sanitize_uploaded_filename(filename: str | None) -> str:
    """Return a safe leaf filename for staging under ``/inputs/uploads``."""

    original = Path(filename or "upload.txt").name
    if original in {"", ".", ".."}:
        original = "upload.txt"

    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", original).lstrip(".")
    if not sanitized:
        sanitized = "upload.txt"
    return sanitized


def _dedupe_staged_name(filename: str, taken_names: set[str]) -> str:
    stem = Path(filename).stem or "upload"
    suffix = Path(filename).suffix
    candidate = filename
    index = 2

    while candidate in taken_names:
        candidate = f"{stem}_{index}{suffix}"
        index += 1

    taken_names.add(candidate)
    return candidate


async def _read_upload_bounded(
    upload: UploadFile, filename: str, max_bytes: int = MAX_ATTACHMENT_BYTES,
    remaining_bytes: int | None = None,
) -> bytes:
    """Read *upload* into memory, refusing anything over the per-file limit.

    Starlette reports ``size`` for multipart uploads, so the common oversized
    case is rejected without reading at all. The chunked loop is the fallback
    for streams that do not report a size: it stops one chunk past the limit
    instead of buffering the entire body first.
    """

    declared_size = getattr(upload, "size", None)
    if isinstance(declared_size, int) and declared_size > max_bytes:
        raise ValueError(
            f"File '{filename}' exceeds the {max_bytes / 1024**2:g} MiB per-file limit."
        )
    if isinstance(declared_size, int) and remaining_bytes is not None and declared_size > remaining_bytes:
        raise ValueError("Attachments exceed the combined upload limit.")

    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(
                f"File '{filename}' exceeds the {max_bytes / 1024**2:g} MiB per-file limit."
            )
        if remaining_bytes is not None and total > remaining_bytes:
            raise ValueError("Attachments exceed the combined upload limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_upload_text(raw: bytes, filename: str) -> str:
    # Strict decoding is what actually distinguishes text from binary. Decoding
    # with errors="replace" and then rejecting U+FFFD also rejected legitimate
    # UTF-8 files that genuinely contain a replacement character.
    #
    # "utf-8-sig" strips a byte-order mark a Windows editor may have written.
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File '{filename}' must be a valid UTF-8 text file.") from exc

    if "\x00" in decoded:
        raise ValueError(f"File '{filename}' must be a valid UTF-8 text file.")
    return decoded


async def normalize_uploaded_attachments(
    attachments: Sequence[UploadFile] | None,
    *,
    existing_upload_paths: Sequence[str] | None = None,
    allowed_suffixes: Collection[str] | None = None,
    preserve_bytes: bool = False,
    upload_policy: UploadPolicy | None = None,
    thread_files: ThreadFiles | None = None,
) -> tuple[list[UploadedAttachment], dict[str, Any]]:
    """Validate a complete batch before publishing text state or workspace bytes.

    ``allowed_suffixes`` retains the legacy application-specific text override.
    Binary uploads are opt-in and never enter graph file state.
    """

    uploads = list(attachments or [])
    files_update: dict[str, Any] = {}
    normalized: list[UploadedAttachment] = []
    taken_names = {
        Path(path).name
        for path in (existing_upload_paths or [])
        if isinstance(path, str) and path.startswith(_UPLOADS_PREFIX)
    }

    try:
        if upload_policy is not None and allowed_suffixes is not None:
            raise ValueError("Specify upload_policy or allowed_suffixes, not both")
        policy = upload_policy or UploadPolicy(
            **({"text_suffixes": frozenset(allowed_suffixes)} if allowed_suffixes is not None else {})
        )
        errors = validate_attachments(
            [(u.filename or "upload.txt", getattr(u, "size", None)) for u in uploads],
            upload_policy=policy,
        )
        if errors:
            raise ValueError(" ".join(errors))
        payloads: list[tuple[str, bytes]] = []
        total = 0
        for index, upload in enumerate(uploads, start=1):
            original_name = upload.filename or f"upload_{index}.txt"
            sanitized_name = sanitize_uploaded_filename(original_name)
            staged_name = _dedupe_staged_name(sanitized_name, taken_names)
            binary = Path(sanitized_name).suffix.lower() in policy.binary_suffixes
            if binary and thread_files is None:
                raise ValueError("Binary uploads require a conversation filesystem")
            raw = await _read_upload_bounded(
                upload, original_name, policy.file_limit(sanitized_name), policy.max_total_bytes - total,
            )
            total += len(raw)
            text = None
            if binary:
                classic = raw.startswith((b"II\x2a\x00", b"MM\x00\x2a")) and len(raw) >= 8
                big = raw.startswith((b"II\x2b\x00\x08\x00\x00\x00", b"MM\x00\x2b\x00\x08\x00\x00")) and len(raw) >= 16
                if not (classic or big):
                    raise ValueError(f"File '{original_name}' has an invalid TIFF header.")
            else:
                text = _decode_upload_text(raw, original_name)
                if preserve_bytes:
                    text = raw.decode("utf-8")
            staged_path = f"{_UPLOADS_PREFIX}{staged_name}"
            if thread_files is None:
                files_update[staged_path] = create_file_data(text)
            payloads.append((staged_name, raw))
            normalized.append(UploadedAttachment(
                original_filename=original_name, staged_path=staged_path,
                char_count=len(text) if text is not None else None,
                size_bytes=len(raw), media_type="image/tiff" if binary else "text/plain",
            ))
        if thread_files is not None and payloads:
            async with thread_files.aguard():
                paths = thread_files.add_uploads(payloads)
            normalized = [replace(upload, staged_path=path) for upload, path in zip(normalized, paths)]
        return normalized, files_update
    finally:
        for upload in uploads:
            await upload.close()


async def get_existing_input_files(
    agent: CompiledStateGraph,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Return the current staged ``/inputs`` files for the active thread."""

    try:
        state: Any = await agent.aget_state(config=config)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to inspect existing staged inputs: %s", exc)
        return {}

    values = getattr(state, "values", {}) or {}
    files = values.get("files", {}) or {}
    return {
        path: file_data
        for path, file_data in files.items()
        if isinstance(path, str) and path.startswith(_INPUTS_PREFIX)
    }


async def prepare_code_chat_turn_inputs(
    agent: CompiledStateGraph,
    config: RunnableConfig,
    message: str,
    attachments: Sequence[UploadFile] | None = None,
    *,
    closing_note: str,
    allowed_suffixes: Collection[str] | None = None,
    thread_files: ThreadFiles | None = None,
    upload_policy: UploadPolicy | None = None,
) -> PreparedCodeChatInputs | None:
    """Prepare staged ``/inputs`` files and a manifest for one chat turn."""

    existing_files = await get_existing_input_files(agent, config)
    existing_upload_paths = sorted(
        path for path in existing_files if path.startswith(_UPLOADS_PREFIX)
    )
    files_update: dict[str, Any | None] = {
        path: None for path in existing_files if not _is_persistent_input_path(path)
    }

    if thread_files is not None:
        async with thread_files.aguard():
            thread_files.migrate(existing_files)
    uploaded, uploaded_updates = await normalize_uploaded_attachments(
        attachments,
        existing_upload_paths=existing_upload_paths,
        allowed_suffixes=allowed_suffixes,
        preserve_bytes=thread_files is not None,
        upload_policy=upload_policy,
        thread_files=thread_files,
    )
    files_update.update(uploaded_updates)
    if thread_files is not None:
        async with thread_files.aguard():
            upload_dir = thread_files.path("/workspace/uploads")
            existing_upload_paths = sorted(
                f"/workspace/uploads/{p.name}" for p in upload_dir.iterdir()
                if p.is_file() and not p.is_symlink()
            ) if upload_dir.is_dir() else []
        # Disk is authoritative. Legacy paths remain readable via the alias map.
        for path in list(files_update):
            if path.startswith(_UPLOADS_PREFIX):
                del files_update[path]
        for path in existing_files:
            if path.startswith(_UPLOADS_PREFIX):
                files_update[path] = None
    has_thread_uploads = bool(existing_upload_paths or uploaded)

    uploads_manifest = build_uploads_manifest(existing_files, existing_upload_paths, uploaded)
    if thread_files is not None and existing_upload_paths:
        uploads_manifest = ("# Conversation uploads\n\n" + "\n".join(
            f"- `{path}`" for path in existing_upload_paths
        ), None)
    if uploads_manifest is not None:
        manifest_content, manifest_created_at = uploads_manifest
        files_update[_UPLOADS_MANIFEST_PATH] = create_file_data(
            manifest_content, created_at=manifest_created_at
        )

    pasted_context = extract_pasted_code_context(message)
    has_turn_inputs = bool(uploaded or pasted_context is not None)
    if has_turn_inputs:
        files_update["/inputs/current_message.txt"] = create_file_data(message)
        if pasted_context and pasted_context.contains_code:
            files_update[staged_code_path(pasted_context)] = create_file_data(pasted_context.code)
        if pasted_context and pasted_context.contains_error:
            files_update["/inputs/current_error.txt"] = create_file_data(
                pasted_context.error_text
            )

    # Materialise exactly the state update the graph is about to receive. This
    # has to happen after the manifest and turn-scoped helper files are added;
    # computing it earlier leaves a process-backed workspace with only the raw
    # upload even though the graph can also see current_message, extracted code
    # or errors, and the upload manifest.
    workspace_files = dict(existing_files)
    for path, file_data in files_update.items():
        if file_data is None:
            workspace_files.pop(path, None)
        else:
            workspace_files[path] = file_data

    if has_turn_inputs or has_thread_uploads:
        return PreparedCodeChatInputs(
            message_override=build_inputs_manifest(
                message,
                uploaded,
                pasted_context,
                has_thread_uploads=has_thread_uploads,
                closing_note=closing_note,
                uploads_prefix="/workspace/uploads/" if thread_files else _UPLOADS_PREFIX,
            ),
            files_update=files_update,
            attachments=tuple(uploaded),
            workspace_files=workspace_files,
        )

    if files_update:
        return PreparedCodeChatInputs(
            message_override=message,
            files_update=files_update,
            attachments=tuple(uploaded),
            workspace_files=workspace_files,
        )

    return None
