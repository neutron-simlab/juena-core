"""The one-line descriptions of tool calls shown in the activity log."""

from __future__ import annotations

from juena_core.server.streaming.tool_labels import (
    LABEL_CHARS,
    code,
    describe_tool_call,
    quote,
)


def test_core_tools_read_as_actions() -> None:
    assert describe_tool_call("read_file", {"file_path": "/inputs/data.dat"}) == (
        "Read `/inputs/data.dat`"
    )
    assert describe_tool_call("execute", {"command": "python fit.py"}) == "Run `python fit.py`"
    assert describe_tool_call(
        "task", {"description": "Fit the curve", "subagent_type": "software-specialist"}
    ) == "Ask the software specialist: “Fit the curve”"


def test_a_missing_argument_leaves_no_dangling_punctuation() -> None:
    assert describe_tool_call("task", {}) == "Ask the specialist"
    assert describe_tool_call("read_file", {}) == "Read"


def test_model_text_cannot_turn_into_formatting() -> None:
    """Streamlit renders "$5 to $10" as LaTeX and "*args" as italics."""
    assert quote("cost $5 to $10 for *args") == r"“cost \$5 to \$10 for \*args”"
    # A backtick would close the code span early.
    assert code("echo `date`") == "`echo 'date'`"


def test_a_long_argument_is_cut_to_one_line() -> None:
    command = "python -c '\n" + "x = 1\n" * 50 + "'"

    label = describe_tool_call("execute", {"command": command})

    assert "\n" not in label
    assert label.endswith("…`")
    assert len(label) <= len("Run ``") + LABEL_CHARS + 1


def test_an_application_label_wins_over_cores() -> None:
    labels = {"execute": lambda args: f"Run in the sandbox: {code(args.get('command'))}"}

    assert describe_tool_call("execute", {"command": "ls"}, labels) == "Run in the sandbox: `ls`"


def test_an_undescribed_tool_shows_its_name_and_first_argument() -> None:
    assert describe_tool_call("tavily_search", {"query": "FRM II flux", "max": 5}) == (
        "tavily_search(FRM II flux)"
    )
    assert describe_tool_call("list_research", {}) == "list_research"
    # Models sometimes send arguments that are not an object at all.
    assert describe_tool_call("list_research", "oops") == "list_research"
