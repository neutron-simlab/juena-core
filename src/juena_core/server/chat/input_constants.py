"""Stub for 01/CP4. Ported from ``juena/server/chat/input_constants.py``
(00-BOUNDARY.md, *Moves whole*).

**Do not rename ``DISPLAY_TEXT_KEY``** (00-BOUNDARY.md, *Two literals*) — it
is stored on real, already-persisted messages. Change the comment explaining
why the name looks wrong; keep the string.
"""

from __future__ import annotations

import re

__all__ = [
    "INPUTS_PREFIX",
    "DISPLAY_TEXT_KEY",
    "DISPLAY_ATTACHMENTS_KEY",
    "UPLOADS_PREFIX",
    "UPLOADS_MANIFEST_PATH",
    "FENCED_CODE_RE",
    "ERROR_LINE_RE",
    "LANGUAGE_SUFFIXES",
]

INPUTS_PREFIX = "/inputs/"
DISPLAY_TEXT_KEY = "juena_display_text"
DISPLAY_ATTACHMENTS_KEY = "juena_attachments"
UPLOADS_PREFIX = "/inputs/uploads/"
UPLOADS_MANIFEST_PATH = "/inputs/uploads_manifest.md"
FENCED_CODE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_#+.-]*)\n(?P<body>.*?)```", re.DOTALL)
ERROR_LINE_RE = re.compile(r"")
LANGUAGE_SUFFIXES: dict[str, str] = {}
