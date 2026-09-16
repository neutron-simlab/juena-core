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
