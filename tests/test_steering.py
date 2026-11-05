"""Real graph steps and response cancellation, without a model provider."""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from juena_core.agents.steering import SteeringMiddleware
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext
from juena_core.server.run_control import RunControl, RunRegistry
from juena_core.server.streaming.own_task import run_in_own_task
from juena_core.agents.delegation import with_delegation_boundary, with_user_message
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.backends import StateBackend


async def controlled_stream(source, control):
    async with run_in_own_task(source, control=control) as frames:
        async for frame in frames:
            yield frame


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def control():
    return RunControl(thread_id="thread", user_id="owner")


@asynccontextmanager
async def checkpoint_saver(kind):
    if kind == "memory":
        yield InMemorySaver()
    else:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        dsn = "postgresql://juena_core_test:juena_core_test@127.0.0.1:55432/juena_core_test"
        async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
            await saver.setup()
            yield saver


@pytest.mark.asyncio
async def test_specialist_and_supervisor_independently_receive_ordered_corrections():
    run = control()
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    @tool
    async def search() -> str:
        """Search a deterministic fixture."""
        entered.set()
        await release.wait()
        return "evidence"

    class Inspect(SteeringMiddleware):
        async def awrap_model_call(self, request, handler):
            seen.append((self.recipient, [m.content for m in request.messages]))
            return await super().awrap_model_call(request, handler)

    specialist = create_agent(
        ScriptedModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search", "args": {}, "id": "search-1"}],
                ),
                AIMessage(content="report"),
            ]
        ),
        tools=[search],
        middleware=[Inspect("specialist")],
        checkpointer=InMemorySaver(),
    )
    ctx = RuntimeModelContext(provider="blablador", model="fixture", control=run)
    future = asyncio.create_task(
        specialist.ainvoke(
            {"messages": [HumanMessage("original objective")]},
            {"configurable": {"thread_id": "specialist"}},
            context=ctx,
        )
    )
    await asyncio.wait_for(entered.wait(), 3)
    run.steer("one", "Focus on FAIR")
    run.steer("two", "Summarize existing evidence")
    run.steer("one", "Focus on FAIR")
    release.set()
    result = await asyncio.wait_for(future, 3)
    assert seen[-1][1][-2:] == ["Focus on FAIR", "Summarize existing evidence"]
    assert len([m for m in result["messages"] if m.content == "Focus on FAIR"]) == 1
    supervisor = create_agent(
        ScriptedModel(responses=[AIMessage(content="answer")]),
        middleware=[Inspect("supervisor")],
    )
    await supervisor.ainvoke(
        {"messages": [HumanMessage("original request")]}, context=ctx
    )
    assert seen[-1] == (
        "supervisor",
        ["original request", "Focus on FAIR", "Summarize existing evidence"],
    )
    assert all(
        len(i["recipients"]) == 2 and i["status"] == "applied"
        for i in run.snapshot()["instructions"]
    )


@pytest.mark.asyncio
async def test_parallel_fresh_graphs_each_receive_the_same_instruction():
    run = control()
    run.steer("one", "Only use existing evidence")
    graphs = [
        create_agent(
            ScriptedModel(responses=[AIMessage(content="report")]),
            middleware=[SteeringMiddleware("specialist")],
        )
        for _ in range(2)
    ]
    ctx = RuntimeModelContext(provider="blablador", model="fixture", control=run)
    results = await asyncio.gather(
        *(
            g.ainvoke({"messages": [HumanMessage("objective")]}, context=ctx)
            for g in graphs
        )
    )
    assert all(
        any(m.content == "Only use existing evidence" for m in r["messages"])
        for r in results
    )


@pytest.mark.asyncio
async def test_real_delegation_inherits_control_without_forwarding_supervisor_state():
    run = control()
    entered, release = asyncio.Event(), asyncio.Event()
    observed = []

    @tool
    async def search() -> str:
        """A blocked specialist research step."""
        entered.set()
        await release.wait()
        return "evidence"

    class Inspect(SteeringMiddleware):
        async def awrap_model_call(self, request, handler):
            observed.append((self.recipient, [m.content for m in request.messages]))
            return await super().awrap_model_call(request, handler)

    specialist = create_agent(
        ScriptedModel(
            responses=[
                AIMessage(
                    content="", tool_calls=[{"name": "search", "args": {}, "id": "s"}]
                ),
                AIMessage(content="report"),
            ]
        ),
        tools=[search],
        middleware=[Inspect("specialist")],
        context_schema=RuntimeModelContext,
    )
    supervisor = create_agent(
        ScriptedModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "subagent_type": "fixture",
                                "description": "original objective",
                            },
                            "id": "delegate",
                        }
                    ],
                ),
                AIMessage(content="answer"),
            ]
        ),
        middleware=[
            Inspect("supervisor"),
            SubAgentMiddleware(
                backend=lambda runtime: StateBackend(runtime),
                subagents=with_delegation_boundary(
                    [
                        {
                            "name": "fixture",
                            "description": "fixture specialist",
                            "runnable": specialist,
                        }
                    ]
                ),
            ),
        ],
        context_schema=RuntimeModelContext,
        checkpointer=InMemorySaver(),
    )
    ctx = RuntimeModelContext(provider="blablador", model="fixture", control=run)
    pending = asyncio.create_task(
        supervisor.ainvoke(
            {"messages": [HumanMessage("original request")]},
            {"configurable": {"thread_id": "delegation"}},
            context=ctx,
        )
    )
    await asyncio.wait_for(entered.wait(), 3)
    run.steer("one", "Summarize the existing evidence")
    release.set()
    await asyncio.wait_for(pending, 3)
    specialist_calls = [messages for role, messages in observed if role == "specialist"]
    supervisor_calls = [messages for role, messages in observed if role == "supervisor"]
    assert specialist_calls[-1][-1] == "Summarize the existing evidence"
    assert supervisor_calls[-1][-1] == "Summarize the existing evidence"
    assert "original request" not in specialist_calls[0]


@pytest.mark.asyncio
async def test_late_instruction_is_recorded_once_as_not_applied():
    run = control()
    graph = create_agent(
        ScriptedModel(responses=[AIMessage(content="answer")]),
        middleware=[SteeringMiddleware("supervisor")],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "thread"}}
    await graph.ainvoke({"messages": [HumanMessage("original")]}, config)
    run.agent, run.config = graph, config

    async def source():
        run.steer("late", "A late correction")
        yield json.dumps({"type": "token", "content": "answer"})

    frames = [json.loads(frame) async for frame in controlled_stream(source(), run)]
    assert frames[-1]["state"] == "completed"
    assert frames[-1]["instructions"][0]["status"] == "not_applied"
    snapshot = await graph.aget_state(config)
    corrections = snapshot.values["unapplied_steering"]
    assert len(corrections) == 1
    assert corrections[0].additional_kwargs["steering"]["status"] == "not_applied"
    with pytest.raises(ValueError):
        run.steer("new", "Do something else")


@pytest.mark.asyncio
async def test_stop_cancels_a_blocked_graph_and_the_next_turn_does_not_resume_it():
    run = control()
    entered, cleaned = asyncio.Event(), asyncio.Event()

    @tool
    async def search() -> str:
        """Wait until cancelled."""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    graph = create_agent(
        ScriptedModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search", "args": {}, "id": "call-1"}],
                ),
                AIMessage(content="fresh answer"),
            ]
        ),
        tools=[search],
        middleware=[SteeringMiddleware("supervisor")],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "thread"}}
    run.agent, run.config = graph, config
    ctx = RuntimeModelContext(provider="blablador", model="fixture", control=run)

    async def source():
        async for _ in graph.astream(
            {"messages": [HumanMessage("original")]}, config, context=ctx
        ):
            yield "{}"

    async def read():
        return [json.loads(frame) async for frame in controlled_stream(source(), run)]

    reader = asyncio.create_task(read())
    await asyncio.wait_for(entered.wait(), 3)
    run.stop()
    frames = await asyncio.wait_for(reader, 3)
    assert cleaned.is_set()
    assert frames[-1]["state"] == "cancelled"
    snapshot = await graph.aget_state(config)
    assert not snapshot.next
    result = await graph.ainvoke(
        {"messages": [HumanMessage("new request")]},
        config,
        context=replace(ctx, control=None),
    )
    assert result["messages"][-1].content == "fresh answer"


@pytest.mark.asyncio
async def test_run_identity_and_duplicate_instruction_isolation():
    registry = RunRegistry()
    run = registry.create(thread_id="thread", user_id="owner")
    assert registry.get(run.id, "thread", "other") is None
    assert registry.get(run.id, "other", "owner") is None
    run.steer("one", "first")
    with pytest.raises(ValueError):
        run.steer("one", "different")
    run.stop()
    with pytest.raises(asyncio.CancelledError):
        with run.operation():
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("saver_kind", ["memory", "postgres"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("kind", ["clarification", "execute_approval"])
@pytest.mark.parametrize("late", [False, True])
async def test_steering_preserves_interrupt_identity_and_resume(
    kind, late, nested, saver_kind
):
    async with checkpoint_saver(saver_kind) as saver:
        await _assert_pause_resume(kind, late, nested, saver)


async def _assert_pause_resume(kind, late, nested, saver):
    run = control()
    calls = []

    @tool
    def ask() -> str:
        """Pause for a clarification or command approval."""
        calls.append("ask")
        if kind == "clarification":
            return interrupt({"kind": kind, "question": "Continue?"})
        return "command completed"

    class MidDelegationSteering(SteeringMiddleware):
        def before_model(self, state, runtime):
            if nested and not late and not run.instructions:
                run.steer("correction", "Use log scale")
            return super().before_model(state, runtime)

    middleware = [MidDelegationSteering("specialist" if nested else "supervisor")]
    if kind == "execute_approval":
        middleware.append(HumanInTheLoopMiddleware(interrupt_on={"ask": True}))
    specialist = create_agent(
        ScriptedModel(
            responses=[
                AIMessage(
                    content="", tool_calls=[{"name": "ask", "args": {}, "id": "ask-1"}]
                ),
                AIMessage(content="resumed"),
            ]
        ),
        tools=[ask],
        middleware=middleware,
        checkpointer=None if nested else saver,
    )
    graph = specialist
    if nested:
        graph = create_agent(
            ScriptedModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "subagent_type": "fixture",
                                    "description": "original objective",
                                },
                                "id": "delegate",
                            }
                        ],
                    ),
                    AIMessage(content="resumed"),
                ]
            ),
            middleware=[
                SteeringMiddleware("supervisor"),
                SubAgentMiddleware(
                    backend=lambda runtime: StateBackend(runtime),
                    subagents=with_delegation_boundary(
                        [
                            {
                                "name": "fixture",
                                "description": "fixture specialist",
                                "runnable": specialist,
                            }
                        ]
                    ),
                ),
            ],
            checkpointer=saver,
        )
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    ctx = RuntimeModelContext(provider="blablador", model="fixture", control=run)
    run.agent, run.config = graph, config
    if not late and not nested:
        run.steer("correction", "Use log scale")
    await graph.ainvoke({"messages": [HumanMessage("original")]}, config, context=ctx)
    before = await graph.aget_state(config)
    saved_before = await saver.aget_tuple(config)
    parent_before = await saver.aget(saved_before.parent_config)
    if late:
        run.steer("correction", "Use log scale")

    await run.finalize(completed=True)
    after = await graph.aget_state(config)
    assert run.state == "paused"
    assert after.interrupts == before.interrupts
    assert [(t.id, t.name) for t in after.tasks] == [
        (t.id, t.name) for t in before.tasks
    ]
    assert calls == (["ask"] if kind == "clarification" else [])
    saved_after = await saver.aget_tuple(config)
    assert saved_after.pending_writes == saved_before.pending_writes
    assert saved_after.parent_config == saved_before.parent_config
    assert await saver.aget(saved_before.parent_config) == parent_before

    answer = "yes" if kind == "clarification" else {"decisions": [{"type": "approve"}]}
    resumed = await graph.ainvoke(
        Command(resume={before.interrupts[0].id: answer}),
        config,
        context=replace(ctx, control=control()),
    )
    assert resumed["messages"][-1].content == "resumed"
    pending_calls = set()
    for message in resumed["messages"]:
        if pending_calls:
            assert isinstance(message, ToolMessage), (
                "Steering split a tool call from its result"
            )
        if isinstance(message, ToolMessage):
            pending_calls.discard(message.tool_call_id)
        elif isinstance(message, AIMessage):
            pending_calls.update(call["id"] for call in message.tool_calls)
    assert not pending_calls
    assert not (await graph.aget_state(config)).interrupts
    await saver.adelete_thread(thread_id)


def test_delegation_attaches_original_request_instead_of_steering():
    original = "Plot the attached specification with all three datasets."
    steering = HumanMessage(
        "Use log scale", additional_kwargs={"steering": {"status": "applied"}}
    )
    brief = with_user_message(
        "Plot the specification", [HumanMessage(original), steering]
    )
    assert original in brief
    assert "Use log scale" not in brief


@pytest.mark.asyncio
async def test_unapplied_instruction_is_history_only_until_explicitly_resubmitted(
    monkeypatch, configured
):
    from juena_core.server.chat import repository

    seen = []

    class Inspect(ScriptedModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append([m.content for m in messages])
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

    saver = InMemorySaver()
    monkeypatch.setattr(repository, "get_checkpointer", lambda: saver)
    graph = create_agent(
        Inspect(responses=[AIMessage(content="answer")]),
        middleware=[SteeringMiddleware("supervisor")],
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": "thread"}}
    await graph.ainvoke({"messages": [HumanMessage("original")]}, config)
    run = control()
    run.agent, run.config = graph, config
    run.steer("late", "Use log scale")
    await run.finalize(completed=True)
    history = await repository.load_thread_messages("thread")
    assert [m.content for m in history] == ["original", "answer", "Use log scale"]
    assert history[-1].custom_data["steering"]["status"] == "not_applied"

    await graph.ainvoke({"messages": [HumanMessage("Explain the data")]}, config)
    assert "Use log scale" not in seen[-1]
    history = await repository.load_thread_messages("thread")
    assert [m.content for m in history][:4] == [
        "original",
        "answer",
        "Use log scale",
        "Explain the data",
    ]

    await graph.ainvoke({"messages": [HumanMessage("Use log scale")]}, config)
    assert seen[-1].count("Use log scale") == 1


@pytest.mark.asyncio
async def test_checkpoint_failure_does_not_claim_sandbox_cleanup_failed(monkeypatch):
    run = control()

    async def fail(_control):
        raise RuntimeError("temporary checkpoint failure")

    monkeypatch.setattr("juena_core.server.run_control._reconcile", fail)
    await run.finalize(completed=True)
    assert run.state == "failed"
    assert not run.cleanup_failed
    assert "history" in run.error.lower()


@pytest.mark.asyncio
async def test_unconfirmed_sandbox_cleanup_retains_protection(monkeypatch):
    run = control()

    async def fail(_control):
        raise RuntimeError("container still running")

    monkeypatch.setattr("juena_core.server.run_control._cancel_sandbox", fail)
    await run.finalize(completed=False)
    assert run.state == "failed"
    assert run.cleanup_failed
