"""Stub for 01/CP4. Ported from ``juena/server/chat/inputs.py``
(00-BOUNDARY.md, *Moves whole*). ``_decode_upload_text`` decodes with
``"utf-8-sig"``, stripping a BOM a Windows editor may have written."""

from __future__ import annotations

from typing import Any

from fastapi import UploadFile

__all__ = [
    "sanitize_uploaded_filename",
    "normalize_uploaded_attachments",
    "get_existing_input_files",
    "prepare_code_chat_turn_inputs",
]


def sanitize_uploaded_filename(filename: str | None) -> str:
    raise NotImplementedError("juena_core.server.chat.inputs.sanitize_uploaded_filename lands in 01/CP4")


def _dedupe_staged_name(filename: str, taken_names: set[str]) -> str:
    raise NotImplementedError("juena_core.server.chat.inputs._dedupe_staged_name lands in 01/CP4")


async def _read_upload_bounded(upload: UploadFile, filename: str) -> bytes:
    raise NotImplementedError("juena_core.server.chat.inputs._read_upload_bounded lands in 01/CP4")


def _decode_upload_text(raw: bytes, filename: str) -> str:
    raise NotImplementedError("juena_core.server.chat.inputs._decode_upload_text lands in 01/CP4")


async def normalize_uploaded_attachments(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.server.chat.inputs.normalize_uploaded_attachments lands in 01/CP4")


async def get_existing_input_files(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.server.chat.inputs.get_existing_input_files lands in 01/CP4")


async def prepare_code_chat_turn_inputs(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.server.chat.inputs.prepare_code_chat_turn_inputs lands in 01/CP4")
