"""Server-side validation for sandbox interrupt decisions."""

from types import SimpleNamespace

import pytest
from langgraph.types import Interrupt

from juena_core.artifacts import ArtifactStore, set_artifact_store_for_tests
from juena_core.sandbox import config as sandbox_config
from juena_core.sandbox.approvals import (
    ApprovalResumeInput,
    register_sandbox_interrupt,
)
from juena_core.sandbox.config import SandboxRuntimeSettings
from juena_core.sandbox.policy import requires_execution_approval
from juena_core.schema.interrupts import ClarificationResumeInput
from juena_core.server import interrupts as interrupts_module
from juena_core.server.interrupts import (
    ResumeError,
    build_resume_command,
    first_pending_interrupt,
    interrupt_event,
)


def _interrupt(interrupt_id: str = "interrupt-1") -> Interrupt:
    return Interrupt(
        id=interrupt_id,
        value={
            "action_requests": [
                {
                    "name": "execute",
                    "args": {"command": "python plot.py", "timeout": 600},
                    "description": "Create the requested plot",
                }
            ],
            "review_configs": [
                {"allowed_decisions": ["approve", "edit", "reject"]}
            ],
        },
    )


class _Agent:
    def __init__(self, interrupts):
        self.snapshot = SimpleNamespace(
            tasks=[SimpleNamespace(interrupts=tuple(interrupts))]
        )

    async def aget_state(self, config):
        return self.snapshot


@pytest.fixture
def artifact_store(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    yield store
    set_artifact_store_for_tests(None)


@pytest.fixture(autouse=True)
def _configured_sandbox(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sandbox_config,
        "_sandbox_settings",
        SandboxRuntimeSettings(
            enabled=True,
            identity_secret="t" * 32,
            workspace_root=tmp_path / "workspaces",
        ),
    )
    isolated_registry = dict(interrupts_module._interrupt_kinds)
    monkeypatch.setattr(interrupts_module, "_interrupt_kinds", isolated_registry)
    register_sandbox_interrupt()
    yield


@pytest.mark.asyncio
async def test_approve_resumes_exact_interrupt(artifact_store) -> None:
    command = await build_resume_command(
        agent=_Agent([_interrupt()]),
        config={},
        user_id="user-a",
        payload=ApprovalResumeInput(
            thread_id="thread-a",
            interrupt_id="interrupt-1",
            decision="approve",
        ),
    )

    assert command.resume == {
        "interrupt-1": {"decisions": [{"type": "approve"}]}
    }


@pytest.mark.asyncio
async def test_edit_changes_only_command_text(artifact_store) -> None:
    command = await build_resume_command(
        agent=_Agent([_interrupt()]),
        config={},
        user_id="user-a",
        payload=ApprovalResumeInput(
            thread_id="thread-a",
            interrupt_id="interrupt-1",
            decision="edit",
            edited_command="python revised.py",
        ),
    )

    [decision] = command.resume["interrupt-1"]["decisions"]
    assert decision["edited_action"] == {
        "name": "execute",
        "args": {"command": "python revised.py", "timeout": 600},
    }


@pytest.mark.asyncio
async def test_reject_sends_a_reject_decision(artifact_store) -> None:
    command = await build_resume_command(
        agent=_Agent([_interrupt()]),
        config={},
        user_id="user-a",
        payload=ApprovalResumeInput(
            thread_id="thread-a",
            interrupt_id="interrupt-1",
            decision="reject",
        ),
    )

    [decision] = command.resume["interrupt-1"]["decisions"]
    assert decision["type"] == "reject"


@pytest.mark.asyncio
async def test_stale_or_cross_thread_interrupt_is_rejected(artifact_store) -> None:
    with pytest.raises(ResumeError, match="stale"):
        await build_resume_command(
            agent=_Agent([_interrupt("different")]),
            config={},
            user_id="user-a",
            payload=ApprovalResumeInput(
                thread_id="thread-a",
                interrupt_id="interrupt-1",
                decision="approve",
            ),
        )


@pytest.mark.asyncio
async def test_non_execute_interrupt_is_rejected(artifact_store) -> None:
    interrupt = _interrupt()
    interrupt.value["action_requests"][0]["name"] = "write_file"
    with pytest.raises(ResumeError, match="not a pending execute_approval"):
        await build_resume_command(
            agent=_Agent([interrupt]),
            config={},
            user_id="user-a",
            payload=ApprovalResumeInput(
                thread_id="thread-a",
                interrupt_id="interrupt-1",
                decision="approve",
            ),
        )


def test_redundant_artifact_export_does_not_request_human_approval() -> None:
    request = SimpleNamespace(
        tool_call={
            "name": "execute",
            "args": {
                "command": (
                    "python -c \"import base64; "
                    "base64.b64encode(open('/workspace/outputs/plot.png','rb').read())\""
                )
            },
        }
    )

    assert requires_execution_approval(request) is False
    request.tool_call["args"]["command"] = "python /workspace/plot.py"
    assert requires_execution_approval(request) is True


def test_approval_card_reports_the_configured_worker_limits(monkeypatch, tmp_path) -> None:
    """The card must describe what the worker will enforce, not a fixed pair.

    Deliberately *not* the default 2 vCPU / 4 GB: an assertion against the
    defaults passes whether the card reads the configuration or hardcodes it,
    and this is the defect that shipped -- the development launcher ran the
    worker at 1 vCPU / 2 GB while the card promised twice that, because the
    API was never given the same environment.
    """

    monkeypatch.setattr(
        sandbox_config,
        "_sandbox_settings",
        SandboxRuntimeSettings(
            enabled=True,
            identity_secret="t" * 32,
            workspace_root=tmp_path / "workspaces",
            cpu_limit="1",
            memory_limit="2g",
        ),
    )

    payload = interrupt_event(_interrupt())

    assert payload is not None
    assert payload["type"] == "approval_required"
    assert payload["limits"] == {
        "timeout_seconds": 600,
        "network": "none",
        "cpu": "1",
        "memory": "2g",
    }


def _question(interrupt_id: str = "interrupt-2") -> Interrupt:
    return Interrupt(
        id=interrupt_id,
        value={
            "kind": "clarification",
            "asked_by": "software-specialist",
            "question": "Which background subtraction should I use?",
            "options": ["Solvent-only", "Empty cell"],
        },
    )


@pytest.mark.asyncio
async def test_answer_resumes_the_asking_agent_with_the_users_own_words(
    artifact_store,
) -> None:
    command = await build_resume_command(
        agent=_Agent([_question()]),
        config={},
        user_id="user-a",
        payload=ClarificationResumeInput(
            kind="clarification",
            thread_id="thread-a",
            interrupt_id="interrupt-2",
            answer="  Empty cell  ",
        ),
    )

    assert command.resume == {"interrupt-2": "Empty cell"}


@pytest.mark.asyncio
async def test_an_answer_cannot_approve_a_pending_sandbox_command(artifact_store) -> None:
    with pytest.raises(ResumeError, match="not a pending clarification"):
        await build_resume_command(
            agent=_Agent([_interrupt()]),
            config={},
            user_id="user-a",
            payload=ClarificationResumeInput(
                kind="clarification",
                thread_id="thread-a",
                interrupt_id="interrupt-1",
                answer="approve it",
            ),
        )


@pytest.mark.asyncio
async def test_parallel_commands_pause_and_resume_the_same_specialist(artifact_store) -> None:
    """Exercise the real HITL payload that used to disappear from the UI."""
    import json

    from deepagents.backends import StateBackend
    from deepagents.middleware.subagents import SubAgentMiddleware
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    from juena_core.clients.base import BaseAgentClient
    from juena_core.server.streaming.processor import StreamEventProcessor

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    executed = []

    @tool
    def execute(command: str, timeout: int = 600) -> str:
        """Record a command without executing anything."""
        executed.append((command, timeout))
        return "ok"

    specialist = create_agent(
        model=ToolModel(responses=[
            AIMessage(content="", tool_calls=[
                {"name": "execute", "args": {"command": f"command-{i}", "timeout": 20 + i}, "id": f"call-{i}"}
                for i in range(3)
            ]),
            AIMessage(content="specialist finished"),
        ]),
        tools=[execute],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"execute": True})],
    )
    agent = create_agent(
        model=ToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "task", "id": "delegate",
                "args": {"description": "inspect", "subagent_type": "specialist"},
            }]),
            AIMessage(content="final answer"),
        ]),
        middleware=[SubAgentMiddleware(backend=StateBackend(), subagents=[{
            "name": "specialist", "description": "test", "runnable": specialist,
        }])],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "batch-approval"}}
    processor = StreamEventProcessor(agent, config, "run-1", "inspect")
    events = []
    async for event in agent.astream(
        {"messages": [HumanMessage(content="inspect")]}, config,
        stream_mode=["updates", "messages"], subgraphs=True, version="v2",
    ):
        events.extend([item async for item in processor.process_event(event)])

    assert executed == []
    [card] = [item for item in events if item["type"] == "approval_required"]
    assert [action["command"] for action in card["actions"]] == [f"command-{i}" for i in range(3)]
    pending, kind = await first_pending_interrupt(agent, config)
    assert kind == "execute_approval"
    assert interrupt_event(pending) == card
    client = BaseAgentClient(base_url="http://unused.invalid", agent="test")
    try:
        assert client._parse_sse_data(json.dumps(card)) == card
    finally:
        client.close()

    command = await build_resume_command(
        agent=agent, config=config, user_id="user-a",
        payload=ApprovalResumeInput(
            thread_id="batch-approval", interrupt_id=pending.id,
            decisions=[
                {"decision": "approve"},
                {"decision": "edit", "edited_command": "revised"},
                {"decision": "reject"},
            ],
        ),
    )
    result = await agent.ainvoke(command, config)
    assert sorted(executed) == [("command-0", 20), ("revised", 21)]
    assert result["messages"][-1].content == "final answer"
    assert await first_pending_interrupt(agent, config) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("decisions", [None, [{"decision": "approve"}]])
async def test_batch_requires_a_decision_for_every_command(artifact_store, decisions) -> None:
    pending = _interrupt()
    pending.value["action_requests"] *= 2
    pending.value["review_configs"] *= 2
    fields = {"decision": "approve"} if decisions is None else {"decisions": decisions}
    with pytest.raises(ResumeError, match="decision for every"):
        await build_resume_command(
            agent=_Agent([pending]), config={}, user_id="user-a",
            payload=ApprovalResumeInput(thread_id="t", interrupt_id=pending.id, **fields),
        )
