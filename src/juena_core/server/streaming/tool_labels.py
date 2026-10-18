"""One short, readable line for each tool call, for the activity log.

The UI lists the steps a model takes -- "Read `/inputs/data.dat`", "Ask the
software specialist: “Fit the curve”" -- the way Claude Code does, rather than
a tool name and a slice of its output. The line is written here, on the
server, from the call's arguments, so a large argument (a whole script, a
file's content) never travels to the browser.

Core describes the tools it installs: the filesystem tools, ``task`` and
``ask_user``. An application describes its own through
``StreamPolicy.tool_labels``. A tool nobody describes is shown as its name and
first argument, e.g. ``tavily_search(neutron flux of FRM II)``.

Lines are Markdown. Use :func:`code` for paths, patterns and commands and
:func:`quote` for prose a model wrote, so the text cannot turn into formatting.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "LABEL_CHARS",
    "ToolLabel",
    "DEFAULT_TOOL_LABELS",
    "short",
    "code",
    "quote",
    "describe",
    "describe_tool_call",
]

#: Longest argument shown in a line. Enough for a path or a query, short
#: enough that a line stays a line.
LABEL_CHARS = 70

#: Turns a call's arguments into its line.
ToolLabel = Callable[[dict[str, Any]], str]

# Characters Markdown (and Streamlit's LaTeX) would read as formatting.
_MARKDOWN_CHARS = re.compile(r"([\\`*_{}\[\]<>#|~$])")


def short(value: Any, limit: int = LABEL_CHARS) -> str:
    """*value* on one line, cut to *limit* characters."""

    text = " ".join(str(value if value is not None else "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def code(value: Any) -> str:
    """*value* as inline code, or "" when there is nothing to show.

    A backtick inside would close the code span early, so it becomes a quote.
    """

    text = short(value).replace("`", "'")
    return f"`{text}`" if text else ""


def quote(value: Any) -> str:
    """*value* in quotation marks, or "" when there is nothing to show.

    Escaped, because a query such as "cost $5 to $10" would otherwise render
    as LaTeX and "*args" as italics.
    """

    text = _MARKDOWN_CHARS.sub(r"\\\1", short(value))
    return f"“{text}”" if text else ""


def describe(action: str, detail: str = "", *, sep: str = " ") -> str:
    """Join an action and its detail, leaving the detail out when it is empty."""

    return f"{action}{sep}{detail}" if detail else action


def _task(args: dict[str, Any]) -> str:
    # "software-specialist" reads as "the software specialist".
    name = short(args.get("subagent_type") or "specialist").replace("-", " ").replace("_", " ")
    return describe(f"Ask the {name}", quote(args.get("description")), sep=": ")


DEFAULT_TOOL_LABELS: dict[str, ToolLabel] = {
    "ls": lambda args: describe("List", code(args.get("path") or "/")),
    "read_file": lambda args: describe("Read", code(args.get("file_path"))),
    "write_file": lambda args: describe("Write", code(args.get("file_path"))),
    "edit_file": lambda args: describe("Edit", code(args.get("file_path"))),
    "delete": lambda args: describe("Delete", code(args.get("file_path"))),
    "glob": lambda args: describe("Find files matching", code(args.get("pattern"))),
    "grep": lambda args: describe("Search files for", code(args.get("pattern"))),
    "execute": lambda args: describe("Run", code(args.get("command"))),
    "task": _task,
    "ask_user": lambda args: describe("Question for you", quote(args.get("question")), sep=": "),
}


def describe_tool_call(
    name: str,
    args: Any,
    labels: Mapping[str, ToolLabel] | None = None,
) -> str:
    """The activity-log line for one tool call.

    The application's ``labels`` win over core's, so an application can
    reword a core tool as well as describe its own.
    """

    args = args if isinstance(args, dict) else {}
    label = (labels or {}).get(name) or DEFAULT_TOOL_LABELS.get(name)
    if label is not None:
        return label(args)

    first = next(
        (value for value in args.values() if isinstance(value, (str, int, float)) and str(value).strip()),
        "",
    )
    detail = _MARKDOWN_CHARS.sub(r"\\\1", short(first))
    return f"{name}({detail})" if detail else name
