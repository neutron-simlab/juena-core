"""The conversation-scoped findings and specialist delegation boundary."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from juena_core.agents.delegation import SpecialistDelegate, with_delegation_boundary
from juena_core.agents.findings import (
    conflict_path,
    crossing_files,
    findings_delta,
    fingerprint,
    validate_findings,
)


def _file(content: str) -> dict[str, str]:
    return {"content": content, "encoding": "utf-8"}


class _Recorder(Runnable):
    def __init__(self, produces: dict | None = None) -> None:
        self.received: dict | None = None
        self._produces = produces or {}

    def invoke(self, input, config=None, **kwargs):  # noqa: A002, ANN001, ANN003
        self.received = input
        return {
            "messages": [AIMessage(content="<specialist_report>ok</specialist_report>")],
            "files": {**input.get("files", {}), **self._produces},
            "execution_events": [{"must_not": "cross"}],
        }

    async def ainvoke(self, input, config=None, **kwargs):  # noqa: A002, ANN001, ANN003
        return self.invoke(input, config, **kwargs)


def test_only_inputs_and_findings_cross_into_a_specialist() -> None:
    """A specialist is told it cannot see the conversation. It could.

    ``SubAgentMiddleware`` copies the parent's whole state in, and a supervisor
    keeps evicted conversation turns under ``/conversation_history/`` in the
    same ``files`` channel that carries findings. So the part of the
    conversation the supervisor had already thrown away was being handed to the
    one agent whose prompt promises it has no access to any of it.

    The state handed in is built from scratch rather than filtered, which is
    why ``future_private_field`` is absent without anyone having named it: a
    denylist would have to be revisited on every upstream release that adds a
    state field.
    """

    inner = _Recorder()
    delegate = SpecialistDelegate(inner, name="specialist")

    delegate.invoke(
        {
            "messages": [HumanMessage("inspect")],
            "files": {
                "/findings/earlier.md": _file("prior evidence"),
                "/inputs/data.csv": _file("q,I"),
                "/conversation_history/thread.md": _file("private history"),
                "/memories/AGENTS.md": _file("saved preference"),
                "/scratch.txt": _file("scratch"),
            },
            "future_private_field": {"secret": True},
        }
    )

    assert set(inner.received or {}) == {"messages", "files"}
    assert set((inner.received or {})["files"]) == {
        "/findings/earlier.md",
        "/inputs/data.csv",
    }


def test_only_the_report_and_changed_findings_cross_back() -> None:
    """The return trip is an allowlist too.

    Deep Agents copies the subagent's whole state back except ``messages``, so
    a scratch file, an execution event, or a finding the run merely inherited
    would all otherwise reach the conversation.
    """

    inner = _Recorder(
        produces={
            "/findings/new.md": _file("new evidence"),
            "/workspace/scratch.py": _file("discarded"),
        }
    )
    result = SpecialistDelegate(inner, name="specialist").invoke(
        {
            "messages": [HumanMessage("inspect")],
            "files": {"/findings/earlier.md": _file("unchanged")},
        }
    )

    assert set(result) == {"messages", "files"}
    assert result["files"] == {"/findings/new.md": _file("new evidence")}
    assert "execution_events" not in result


def test_async_delegation_has_the_same_boundary() -> None:
    """Both entry points are wrapped, not just the one the tests above take."""

    import asyncio

    inner = _Recorder({"/findings/new.md": _file("new")})
    result = asyncio.run(
        SpecialistDelegate(inner, name="specialist").ainvoke(
            {"messages": [HumanMessage("inspect")], "files": {}}
        )
    )

    assert result["files"] == {"/findings/new.md": _file("new")}


def test_registration_wraps_every_specialist() -> None:
    wrapped = with_delegation_boundary(
        [{"name": "a", "description": "d", "runnable": _Recorder()}]
    )

    assert isinstance(wrapped[0]["runnable"], SpecialistDelegate)
    assert wrapped[0]["name"] == "a"


def test_a_specialist_that_already_has_a_boundary_is_not_wrapped_again() -> None:
    """The outer wrapper can only ever remove what the inner one allowed.

    An application may subclass the delegate to let one more field back out of
    its specialists, and it reaches this function through
    `build_supervisor_middleware`. Wrapping that subclass in a plain
    `SpecialistDelegate` would drop the extra field with no error at all.
    """

    class _ExtraField(SpecialistDelegate):
        def _outbound(self, sent, result):
            crossing = super()._outbound(sent, result)
            crossing["module_results"] = {"guide": {"ok": True}}
            return crossing

    delegate = _ExtraField(_Recorder(), name="a")

    wrapped = with_delegation_boundary(
        [{"name": "a", "description": "d", "runnable": delegate}]
    )

    assert wrapped[0]["runnable"] is delegate
    assert wrapped[0]["runnable"].invoke({"messages": [], "files": {}})[
        "module_results"
    ] == {"guide": {"ok": True}}


def test_findings_helpers_return_deltas_and_stable_conflict_paths() -> None:
    before = {"/findings/a.md": _file("old"), "/inputs/data.csv": _file("input")}
    after = {
        "/findings/a.md": _file("changed"),
        "/findings/b.md": _file("new"),
        "/inputs/data.csv": _file("input"),
    }

    assert crossing_files(after) == after
    assert findings_delta(before, after) == {
        "/findings/a.md": _file("changed"),
        "/findings/b.md": _file("new"),
    }
    assert fingerprint(before) == {"/findings/a.md": "old"}
    assert conflict_path("/findings/a.md", "job-1") == (
        "/findings/_conflicts/job-1/a.md"
    )


def test_persisted_findings_are_shape_and_namespace_validated() -> None:
    value = validate_findings(
        {
            "/findings/valid.md": {"content": "ok"},
            "/findings/_conflicts/forged.md": {"content": "no"},
            "/conversation_history/private.md": {"content": "no"},
            "/findings/broken.md": {"content": None},
        }
    )

    assert value == {
        "/findings/valid.md": {"content": "ok", "encoding": "utf-8"}
    }


# The user's message reaches the specialist even when the brief leaves it out.

SPEC = "Write fit.py: read q, I, dI from data.dat, fit a sphere, report R in nm."


def _supervisor_brief(description: str):  # noqa: ANN202
    """Run the real supervisor stack once and return the brief the specialist got.

    The user's turn is built by the server's own input builder, and the supervisor
    model asks for the specialist through the real `task` tool.
    """

    import asyncio
    from uuid import uuid4

    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.store.memory import InMemoryStore

    from juena_core.agents.backends import build_supervisor_backend
    from juena_core.agents.specialist_runtime import build_supervisor_middleware
    from juena_core.server.agent.input_handler import AgentInputHandler

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            return self

    specialist = _Recorder()
    supervisor = create_agent(
        model=ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "task",
                        "args": {"description": description, "subagent_type": "software"},
                        "id": "call-1",
                    }],
                ),
                AIMessage(content="done"),
            ]
        ),
        middleware=build_supervisor_middleware(
            backend=build_supervisor_backend(InMemoryStore()),
            summarizer_model=ToolModel(responses=[AIMessage(content="summary")]),
            fallback_models=[],
            subagents=[{
                "name": "software",
                "description": "Write and check code.",
                "runnable": specialist,
            }],
        ),
    )
    ids = {"thread_id": str(uuid4()), "user_id": str(uuid4())}
    state = asyncio.run(supervisor.ainvoke(
        AgentInputHandler.build_input(SPEC),
        {"configurable": ids},
        # No provider or model here, so the fake model above is not swapped out.
        context=ids,
    ))
    brief = specialist.received["messages"][0].content
    written = state["messages"][1].tool_calls[0]["args"]["description"]
    return brief, written


def test_a_brief_without_the_spec_reaches_the_specialist_with_it() -> None:
    """The failure seen in the benchmark: "per the user's spec", and no spec."""

    from juena_core.agents.delegation import USER_MESSAGE_NOTE

    brief, written = _supervisor_brief("Write the script per the user's spec.")

    assert brief.startswith("Write the script per the user's spec.")
    assert USER_MESSAGE_NOTE in brief
    assert f"<user_message>\n{SPEC}\n</user_message>" in brief
    # The supervisor's own history keeps the brief as its model wrote it.
    assert written == "Write the script per the user's spec."


def test_a_brief_that_already_quotes_the_message_is_left_alone() -> None:
    description = f"Do exactly this:\n{SPEC}"

    brief, _ = _supervisor_brief(description)

    assert brief == description


def test_only_the_latest_user_message_is_attached() -> None:
    """An earlier request may have been replaced; carrying it is the supervisor's job."""

    from juena_core.agents.delegation import with_user_message
    from juena_core.server.agent.input_handler import AgentInputHandler

    messages = [
        *AgentInputHandler.build_input("Plot it, but do not run anything.")["messages"],
        AIMessage(content="Here is the plan."),
        *AgentInputHandler.build_input(SPEC)["messages"],
    ]

    brief = with_user_message("Write the script.", messages)

    assert SPEC in brief
    assert "do not run anything" not in brief
