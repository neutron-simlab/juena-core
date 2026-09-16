"""Approval records written around sandbox tool execution."""

import io
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from PIL import Image

from juena_core.artifacts import ArtifactStore, set_artifact_store_for_tests
from juena_core.sandbox.middleware import SandboxExecutionMiddleware
from juena_core.sandbox.evidence import record_sandbox_execution


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def store(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    yield store
    set_artifact_store_for_tests(None)


def _request(command: str, *, name: str = "execute", user_id: str = "user-a"):
    return SimpleNamespace(
        tool_call={"name": name, "args": {"command": command}, "id": "call-1"},
        runtime=SimpleNamespace(
            context=SimpleNamespace(user_id=user_id, thread_id="thread-a"),
            config={"run_id": "run-1"},
        ),
    )


def _handler(
    text: str = "plot written\n",
    *,
    status: str = "completed",
    exit_code: int | None = 0,
):
    def handle(request):
        record_sandbox_execution(
            command=request.tool_call["args"]["command"],
            status=status,
            exit_code=exit_code,
        )
        return ToolMessage(content=text, tool_call_id="call-1")

    return handle


def _message(result) -> ToolMessage:
    if isinstance(result, ToolMessage):
        return result
    assert isinstance(result, Command)
    return next(
        item for item in result.update["messages"] if isinstance(item, ToolMessage)
    )


def _events(result) -> list[dict]:
    assert isinstance(result, Command)
    return result.update["execution_events"]


def _filenames(store) -> list[str]:
    return [item["filename"] for item in store.claim_for_message("user-a", "thread-a")]


def test_approved_command_and_output_are_recorded(store) -> None:
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(_request("python plot.py"), _handler())

    assert _message(result).text == "plot written\n"
    assert _message(result).status == "success"
    assert result.update["execution_events"][0]["exit_code"] == 0
    assert _filenames(store) == ["generated-command.sh", "execution-output.txt"]


def test_many_approved_commands_leave_room_for_results(store) -> None:
    """Records are the application's bookkeeping and must not spend the user's budget.

    Six approved commands used to write twelve records into a twelve-file
    allowance, so the figure the turn was about had nowhere to go.
    """
    middleware = SandboxExecutionMiddleware()
    for index in range(6):
        middleware.wrap_tool_call(_request(f"python step_{index}.py"), _handler())

    plot = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
        filename="fit.png",
        content=_png(),
    )

    assert plot.kind == "image"


def test_undelivered_write_waits_for_the_specialist_outcome(store) -> None:
    store.note_undelivered("user-a", "thread-a", "report.html", "type is not allowed")
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(_request("python plot.py"), _handler())

    assert _message(result).text == "plot written\n"
    assert store.peek_undelivered("user-a", "thread-a") == [
        ("report.html", "type is not allowed")
    ]
    second = middleware.wrap_tool_call(_request("python again.py"), _handler())
    assert _message(second).text == "plot written\n"
    assert store.peek_undelivered("user-a", "thread-a") == [
        ("report.html", "type is not allowed")
    ]


def test_execution_record_matches_what_the_model_was_shown(store) -> None:
    store.note_undelivered("user-a", "thread-a", "report.html", "type is not allowed")
    middleware = SandboxExecutionMiddleware()

    middleware.wrap_tool_call(_request("python plot.py"), _handler())

    saved = store.claim_for_message("user-a", "thread-a")
    output_ref = next(item for item in saved if item["filename"] == "execution-output.txt")
    _ref, content = store.get("user-a", output_ref["artifact_id"])
    assert content == b"plot written\n"


def test_a_command_with_nothing_queued_is_left_alone(store) -> None:
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(_request("python plot.py"), _handler())

    assert _message(result).text == "plot written\n"


def test_edited_command_is_what_gets_recorded(store) -> None:
    """The case the old exact-text lookup could not handle.

    The human edited the command before approving, so `wrap_tool_call` receives
    the edited text. Nothing is matched against the originally proposed command.
    """
    middleware = SandboxExecutionMiddleware()

    middleware.wrap_tool_call(_request("python revised.py --seed 7"), _handler())

    saved = store.claim_for_message("user-a", "thread-a")
    command_ref = next(item for item in saved if item["filename"] == "generated-command.sh")
    _ref, content = store.get("user-a", command_ref["artifact_id"])
    assert content == b"python revised.py --seed 7\n"


@pytest.mark.asyncio
async def test_async_path_records_the_same_artifacts(store) -> None:
    middleware = SandboxExecutionMiddleware()

    async def handle(request):
        record_sandbox_execution(
            command=request.tool_call["args"]["command"],
            status="completed",
            exit_code=0,
        )
        return ToolMessage(content="done\n", tool_call_id="call-1")

    await middleware.awrap_tool_call(_request("python plot.py"), handle)

    assert _filenames(store) == ["generated-command.sh", "execution-output.txt"]


def test_other_tools_are_not_recorded(store) -> None:
    middleware = SandboxExecutionMiddleware()

    middleware.wrap_tool_call(_request("python plot.py", name="write_file"), _handler())

    assert _filenames(store) == []


def test_redundant_artifact_export_is_not_recorded(store) -> None:
    """Commands that never reach the approval gate get no approval record."""
    middleware = SandboxExecutionMiddleware()

    middleware.wrap_tool_call(
        _request("base64 /workspace/outputs/plot.png"), _handler()
    )

    assert _filenames(store) == []


def test_unauthenticated_run_is_not_recorded(store) -> None:
    middleware = SandboxExecutionMiddleware()
    request = _request("python plot.py")
    request.runtime.context = SimpleNamespace(user_id=None, thread_id=None)

    result = middleware.wrap_tool_call(request, _handler())

    assert _message(result).text == "plot written\n"
    assert _filenames(store) == []


def test_a_failed_run_is_still_evidence_without_an_identity(store) -> None:
    """Audit records need a user id; proving what ran must not.

    Gating both on the identity let a missing user id erase the record of a
    failed command, which downstream reads as "nothing ran" -- fail-open on
    exactly the path that has to fail closed.
    """
    middleware = SandboxExecutionMiddleware()
    request = _request("python fails.py")
    request.runtime.context = SimpleNamespace(user_id=None, thread_id=None)

    result = middleware.wrap_tool_call(
        request, _handler("boom", status="failed", exit_code=1)
    )

    events = _events(result)
    assert [(item["status"], item["exit_code"]) for item in events] == [("failed", 1)]
    assert _message(result).status == "error"
    assert _filenames(store) == []


def test_a_refused_export_is_recorded_but_is_not_a_failed_run(store) -> None:
    """The backend skips these before they run, so they must not read as failures."""
    middleware = SandboxExecutionMiddleware()
    command = "base64 /workspace/outputs/plot.png"

    result = middleware.wrap_tool_call(
        _request(command), _handler("already collected", status="skipped", exit_code=0)
    )

    events = _events(result)
    assert [item["status"] for item in events] == ["skipped"]
    # Guidance the model should act on, not a tool error to retry around.
    assert _message(result).status == "success"


def test_nonzero_execution_is_error_and_preserves_typed_evidence(store) -> None:
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(
        _request("python broken.py"),
        _handler("traceback", status="failed", exit_code=1),
    )

    assert isinstance(result, Command)
    assert _message(result).status == "error"
    assert result.update["execution_events"] == [
        {
            "graph_run_id": "run-1",
            "command": "python broken.py",
            "status": "failed",
            "exit_code": 1,
            "truncated": False,
            "artifact_ids": [],
            "artifact_filenames": [],
            "dropped": [],
        }
    ]


def test_missing_backend_evidence_is_a_tool_error(store) -> None:
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(
        _request("python unknown.py"),
        lambda _request: ToolMessage(content="unknown", tool_call_id="call-1"),
    )

    assert _message(result).status == "error"
    assert result.update["execution_events"][0]["status"] == "tool_error"


def test_tool_still_runs_when_the_record_cannot_be_saved(store) -> None:
    """An unwritable artifact store must not block the command."""
    middleware = SandboxExecutionMiddleware()
    store.root.write_text("not a directory", encoding="utf-8")

    result = middleware.wrap_tool_call(_request("python plot.py"), _handler())

    assert _message(result).text == "plot written\n"


def test_execute_still_runs_where_the_tool_is_actually_bound(store) -> None:
    """A specialist is a subgraph, and a subgraph has no config ``run_id``.

    Every stand-in above hands the middleware a runtime whose ``config``
    carries ``run_id``. A real one does not: ``execute`` is bound inside a
    specialist subagent, LangGraph fills ``execution_info.run_id`` from the
    config of the graph that is running, and a subgraph does not inherit its
    parent's. So the condition this asserts -- a context with the invocation
    id, a config without it -- is the only condition ``execute`` ever runs
    under in the running application, and the generous stand-in is why a
    version that raised there passed the whole suite.

    Built with the real producers: a real ``create_agent`` graph, the real
    middleware, and the real ``RuntimeModelContext`` the server builds.
    """

    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext

    ran: list[str] = []

    @tool
    def execute(command: str) -> str:
        """Run one command in the sandbox."""
        ran.append(command)
        record_sandbox_execution(command=command, status="completed", exit_code=0)
        return "plot written\n"

    class _ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            return self

    agent = create_agent(
        model=_ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "execute", "args": {"command": "python plot.py"}, "id": "c1"}
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        tools=[execute],
        middleware=[SandboxExecutionMiddleware()],
    )

    agent.invoke(
        {"messages": [("user", "plot it")]},
        # No `run_id` in the config, exactly as a subgraph receives it.
        config={"configurable": {"thread_id": "thread-a"}},
        context=RuntimeModelContext(
            provider="blablador",
            model="fake",
            thread_id="thread-a",
            user_id="user-a",
            run_id="run-42",
        ),
    )

    assert ran == ["python plot.py"], "the approved command never reached the sandbox"


def test_evidence_is_scoped_by_the_invocation_id_the_context_carries(store) -> None:
    """The id written into evidence is the one a supervisor can match on.

    `execution_events` is private state, so the graph test above cannot read
    it back; this asserts the same runtime shape one level down, where the
    value is visible.
    """

    middleware = SandboxExecutionMiddleware()
    request = _request("python plot.py")
    # A subgraph's runtime: the invocation id is in the context and nowhere else.
    request.runtime = SimpleNamespace(
        context=SimpleNamespace(user_id="user-a", thread_id="thread-a", run_id="run-42"),
        execution_info=SimpleNamespace(run_id=None),
        config={"configurable": {"thread_id": "thread-a"}},
    )

    result = middleware.wrap_tool_call(request, _handler())

    assert [event["graph_run_id"] for event in _events(result)] == ["run-42"]
    assert [event["exit_code"] for event in _events(result)] == [0]
