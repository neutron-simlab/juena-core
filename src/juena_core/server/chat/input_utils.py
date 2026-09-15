"""Stub for 01/CP4. Ported from ``juena/server/chat/input_utils.py``
(00-BOUNDARY.md, *Moves whole*)."""

from __future__ import annotations

from typing import Any

from juena_core.server.chat.input_types import PastedCodeContext

__all__ = [
    "extract_pasted_code_context",
    "staged_code_path",
    "is_persistent_input_path",
    "build_uploads_manifest",
    "build_inputs_manifest",
]


def _line_is_error_like(line: str) -> bool:
    raise NotImplementedError("juena_core.server.chat.input_utils._line_is_error_like lands in 01/CP4")


def _block_is_error(body: str) -> bool:
    raise NotImplementedError("juena_core.server.chat.input_utils._block_is_error lands in 01/CP4")


def _compact_summary(text: str, *, max_chars: int = 500) -> str:
    raise NotImplementedError("juena_core.server.chat.input_utils._compact_summary lands in 01/CP4")


def extract_pasted_code_context(message: str) -> PastedCodeContext | None:
    raise NotImplementedError("juena_core.server.chat.input_utils.extract_pasted_code_context lands in 01/CP4")


def staged_code_path(context: PastedCodeContext) -> str:
    raise NotImplementedError("juena_core.server.chat.input_utils.staged_code_path lands in 01/CP4")


def is_persistent_input_path(path: str) -> bool:
    raise NotImplementedError("juena_core.server.chat.input_utils.is_persistent_input_path lands in 01/CP4")


def _format_upload_manifest_entry(*args: Any, **kwargs: Any) -> str:
    raise NotImplementedError("juena_core.server.chat.input_utils._format_upload_manifest_entry lands in 01/CP4")


def build_uploads_manifest(*args: Any, **kwargs: Any) -> str:
    raise NotImplementedError("juena_core.server.chat.input_utils.build_uploads_manifest lands in 01/CP4")


def build_inputs_manifest(*args: Any, **kwargs: Any) -> str:
    raise NotImplementedError("juena_core.server.chat.input_utils.build_inputs_manifest lands in 01/CP4")
