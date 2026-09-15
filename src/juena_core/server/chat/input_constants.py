"""Static constants and regexes for staged chat input handling."""

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

# `additional_kwargs` keys on the staged human message. The manifest is what the
# model reads, but the checkpointer is the only store of message content, so the
# text the user actually typed has to ride along or a reloaded chat shows the
# manifest in the user's own bubble.
#
# These two strings are **not renamed** for this package. They are stored on
# real, already-persisted messages in juena-chatbot's checkpointer; renaming
# them would make every existing thread render its manifest in the user's
# bubble. The `juena_` prefix is history, not ownership.
DISPLAY_TEXT_KEY = "juena_display_text"
DISPLAY_ATTACHMENTS_KEY = "juena_attachments"

UPLOADS_PREFIX = "/inputs/uploads/"
UPLOADS_MANIFEST_PATH = "/inputs/uploads_manifest.md"
FENCED_CODE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_#+.-]*)\n(?P<body>.*?)```", re.DOTALL)
# MULTILINE so the `^`-anchored alternatives match at the start of any line, not
# only at the start of the string -- without it, JS/Java stack traces went
# undetected. The trailing bare `Error:` alternative is deliberately absent: it
# matched ordinary prose such as "What does the Error: message mean?".
ERROR_LINE_RE = re.compile(
    r"(Traceback \(most recent call last\):|^\s*File \".*\", line \d+|^\s*at .+\(.+:\d+\)|"
    r"^\s*Caused by:|^\s*ERROR\b|\b[A-Za-z_][A-Za-z0-9_]*(Error|Exception)\b:)",
    re.MULTILINE,
)
LANGUAGE_SUFFIXES = {
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "java": ".java",
    "c": ".c",
    "cpp": ".cpp",
    "c++": ".cpp",
    "rust": ".rs",
    "rs": ".rs",
    "go": ".go",
    "shell": ".sh",
    "bash": ".sh",
    "sh": ".sh",
    "json": ".json",
    "yaml": ".yaml",
    "yml": ".yml",
    "toml": ".toml",
    "sql": ".sql",
    "html": ".html",
    "xml": ".xml",
    "css": ".css",
}
