"""Foreground run identity, steering mailboxes and cancellation ownership.

The API has one process. Only live execution belongs here; messages are copied
to the graph checkpoint by its owner after execution has unwound.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic
from typing import Any
from uuid import uuid4

import anyio
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)
TERMINAL = frozenset({"completed", "paused", "cancelled", "failed"})
__all__ = ["RunControl", "RunRegistry", "control_from"]


@dataclass
class Instruction:
    id: str
    text: str
    recipients: set[str] = field(default_factory=set)

    def status(self, *, final: bool) -> str:
        if self.recipients:
            return "applied"
        return "not_applied" if final else "received"

    def message(self, run_id: str, *, final: bool = False) -> HumanMessage:
        return HumanMessage(
            id=f"steering:{run_id}:{self.id}",
            content=self.text,
            additional_kwargs={
                "steering": {
                    "instruction_id": self.id,
                    "run_id": run_id,
                    "status": self.status(final=final),
                    "recipients": sorted(self.recipients),
                }
            },
        )


class RunControl:
    """One foreground invocation, shared by its supervisor and specialists.

    The response task owns finalization. The lock protects admission and the
    instruction mailbox, including synchronous sandbox tool threads.
    """

    def __init__(self, *, thread_id: str, user_id: str) -> None:
        self.id = str(uuid4())
        self.thread_id, self.user_id = thread_id, user_id
        self.state = "running"
        self.error: str | None = None
        self.instructions: dict[str, Instruction] = {}
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.operations = 0
        self.loop = asyncio.get_running_loop()
        self.notify = None
        self.partial_text = ""
        self.failure: str | None = None
        self.cleanup_failed = False
        self.sandbox_jobs = None
        self.task: asyncio.Task | None = None
        self.agent: Any = None
        self.config: dict | None = None
        self.finished_at: float | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "type": "run",
                "run_id": self.id,
                "thread_id": self.thread_id,
                "state": self.state,
                "error": self.error,
                "instructions": [
                    {
                        "id": item.id,
                        "text": item.text,
                        "status": item.status(final=self.state in TERMINAL),
                        "recipients": sorted(item.recipients),
                    }
                    for item in self.instructions.values()
                ],
            }

    def emit(self) -> None:
        if self.notify is not None:
            self.loop.call_soon_threadsafe(self.notify, self.snapshot())

    def steer(self, instruction_id: str, text: str) -> dict[str, Any]:
        """Accept a correction once; retries cannot change its meaning."""
        with self.lock:
            existing = self.instructions.get(instruction_id)
            if existing is not None:
                if existing.text != text:
                    raise ValueError(
                        "This instruction ID already identifies different text"
                    )
                return self.snapshot()
            if self.state != "running":
                raise ValueError("This response is no longer accepting instructions")
            self.instructions[instruction_id] = Instruction(instruction_id, text)
            logger.info(
                "Steering received run=%s instruction=%s", self.id, instruction_id
            )
            self.emit()
            return self.snapshot()

    def stop(self) -> None:
        """Close admission before asking the response task to unwind."""
        with self.lock:
            if self.state not in {"preparing", "running"}:
                return
            self.state = "stopping"
            self.cancelled.set()
            logger.info("Stop requested run=%s", self.id)
            if self.task is not None:
                self.loop.call_soon_threadsafe(self.task.cancel)
            self.emit()

    def check(self) -> None:
        if self.cancelled.is_set():
            raise asyncio.CancelledError

    @contextmanager
    def operation(self):
        # Register before admission, including admission that is blocked in a
        # synchronous database call. Finalization waits for these operations.
        with self.lock:
            self.check()
            self.operations += 1
        try:
            yield
        finally:
            with self.lock:
                self.operations -= 1

    def messages(
        self, seen: set[str] = frozenset(), *, final: bool = False
    ) -> list[HumanMessage]:
        with self.lock:
            return [
                item.message(self.id, final=final)
                for item in self.instructions.values()
                if item.id not in seen
            ]

    def applied(self, ids: set[str], recipient: str) -> None:
        with self.lock:
            for key in ids:
                item = self.instructions.get(key)
                if item is not None and recipient not in item.recipients:
                    item.recipients.add(recipient)
                    logger.info(
                        "Steering applied run=%s instruction=%s recipient=%s",
                        self.id,
                        key,
                        recipient,
                    )
                    self.emit()

    def finish(self, state: str, error: str | None = None) -> None:
        with self.lock:
            self.state, self.error = state, error
            self.finished_at = monotonic()
            self.emit()
            logger.info("Run finished run=%s state=%s", self.id, state)

    def observe(self, payload: dict) -> None:
        if payload.get("type") == "error":
            self.failure = "The response failed"
        elif payload.get("type") == "token":
            self.partial_text += payload.get("content", "")
        elif (
            payload.get("type") == "message"
            and payload.get("content", {}).get("type") == "ai"
        ):
            self.partial_text = ""

    async def finalize(self, completed: bool) -> None:
        """Confirm termination, then save delivery after all graph tasks exit.

        A history-write failure is reported but does not imply that a container
        remains alive. Only unconfirmed execution cleanup retains the lease.
        """
        with self.lock:
            if not completed:
                self.cancelled.set()
            self.state = "finalizing"
        sandbox_confirmed = not self.cancelled.is_set()
        try:
            with anyio.fail_after(20, shield=True):
                if self.cancelled.is_set():
                    await _cancel_sandbox(self)
                    sandbox_confirmed = True
                await _reconcile(self)
                paused = False
                if completed and self.agent is not None:
                    snapshot = await self.agent.aget_state(self.config)
                    paused = any(t.interrupts for t in snapshot.tasks)
                if self.failure:
                    state = "failed"
                elif self.cancelled.is_set():
                    state = "cancelled"
                elif paused:
                    state = "paused"
                else:
                    state = "completed"
                self.finish(state, self.failure)
        except Exception:
            logger.exception("Run cleanup failed run=%s", self.id)
            self.cleanup_failed = not sandbox_confirmed
            if self.cleanup_failed:
                self.finish(
                    "failed",
                    "Response cleanup failed; sandbox termination may be incomplete",
                )
            else:
                self.finish(
                    "failed", "Response ended, but saving conversation history failed"
                )


class RunRegistry:
    def __init__(self) -> None:
        self.runs: OrderedDict[str, RunControl] = OrderedDict()

    def create(self, *, thread_id: str, user_id: str) -> RunControl:
        now = monotonic()
        for key, run in list(self.runs.items()):
            if run.finished_at is not None and (
                now - run.finished_at > 3600 or len(self.runs) >= 256
            ):
                del self.runs[key]
        run = RunControl(thread_id=thread_id, user_id=user_id)
        self.runs[run.id] = run
        return run

    def get(self, run_id: str, thread_id: str, user_id: str) -> RunControl | None:
        run = self.runs.get(run_id)
        return (
            run
            if run and run.thread_id == thread_id and run.user_id == user_id
            else None
        )


def control_from(context: Any) -> RunControl | None:
    return (
        context.get("control")
        if isinstance(context, dict)
        else getattr(context, "control", None)
    )


async def _reconcile(control: RunControl) -> None:
    """Save delivery receipts without treating a pause as a completed node."""
    if control.agent is None or control.config is None:
        return
    snapshot = await control.agent.aget_state(control.config)
    messages = list(snapshot.values.get("messages", []))
    updates, unapplied = [], []
    for message in control.messages(final=True):
        if message.additional_kwargs["steering"]["status"] == "not_applied":
            # Keep its place in the transcript without making it model input.
            message.additional_kwargs["steering"]["after_message_id"] = (
                messages[-1].id if messages else None
            )
            unapplied.append(message)
        else:
            updates.append(message)
    unapplied_ids = {message.id for message in unapplied}
    # before_model may have appended an instruction just before Stop, without
    # any subsequent model request. Move that message out of model history too.
    updates[:0] = [RemoveMessage(id=m.id) for m in messages if m.id in unapplied_ids]
    if control.cancelled.is_set():
        # Do not leave an unfinished tool call ahead of the next user turn.
        answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        updates = [
            ToolMessage(
                content="Cancelled by the user. Do not resume this action.",
                tool_call_id=call["id"],
            )
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
            if call.get("id") and call["id"] not in answered
        ] + updates
        if control.partial_text:
            updates.append(
                AIMessage(
                    id=f"partial:{control.id}",
                    content=control.partial_text,
                    additional_kwargs={"run_status": "cancelled"},
                )
            )
        updates.append(
            AIMessage(
                id=f"stopped:{control.id}",
                content="Response stopped.",
                additional_kwargs={"run_status": "cancelled"},
            )
        )
    values = {}
    if updates:
        values["messages"] = updates
    if unapplied:
        values["unapplied_steering"] = unapplied
    if control.cancelled.is_set():
        # Stop deliberately discards pending work, unlike a clarification pause.
        await control.agent.aupdate_state(control.config, values, as_node="__start__")
        await control.agent.aupdate_state(control.config, None, as_node="__end__")
    elif values:
        await _save_delivery(control, values)


async def _save_delivery(control: RunControl, updates: dict) -> None:
    """Amend only message channels in the quiescent root checkpoint.

    Graph.update_state simulates a node finishing and changes pending task IDs;
    that invalidates a saved clarification or approval. The saver API lets us
    retain the checkpoint identity, scheduling channels and interrupt writes.
    New channel versions keep earlier checkpoints' message blobs unchanged.
    The foreground lease is still held, so no graph can execute or write here
    concurrently. This boundary is tested with both memory and Postgres savers.
    """
    saver = control.agent.checkpointer
    saved = await saver.aget_tuple(control.config)
    checkpoint = saved.checkpoint.copy()
    checkpoint["channel_values"] = saved.checkpoint["channel_values"].copy()
    checkpoint["channel_versions"] = saved.checkpoint["channel_versions"].copy()
    versions = {}
    for channel, messages in updates.items():
        previous = checkpoint["channel_values"].get(channel, [])
        merged = add_messages(previous, messages)
        if channel == "messages":
            # A specialist may receive steering just before it pauses, while
            # the root still awaits its task result. Put newly copied receipts
            # before that pending call so its future result stays adjacent.
            previous_ids = {message.id for message in previous}
            added = [
                message
                for message in messages
                if isinstance(message, HumanMessage) and message.id not in previous_ids
            ]
            added_ids = {message.id for message in added}
            merged = [message for message in merged if message.id not in added_ids]
            answered = {
                message.tool_call_id
                for message in merged
                if isinstance(message, ToolMessage)
            }
            position = next(
                (
                    index
                    for index, message in enumerate(merged)
                    if isinstance(message, AIMessage)
                    and any(call["id"] not in answered for call in message.tool_calls)
                ),
                len(merged),
            )
            merged[position:position] = added
        checkpoint["channel_values"][channel] = merged
        versions[channel] = saver.get_next_version(
            checkpoint["channel_versions"].get(channel), None
        )
    checkpoint["channel_versions"].update(versions)
    parent = saved.parent_config or {
        "configurable": {
            **saved.config["configurable"],
            "checkpoint_id": None,
        }
    }
    await saver.aput(parent, checkpoint, saved.metadata, versions)


async def _cancel_sandbox(control: RunControl) -> None:
    deadline = monotonic() + 15
    while True:
        jobs = control.sandbox_jobs
        active = (
            await asyncio.to_thread(jobs.cancel_run, control.id)
            if jobs is not None
            else 0
        )
        with control.lock:
            operations = control.operations
        if not active and not operations:
            return
        if monotonic() >= deadline:
            raise RuntimeError(
                "Agent stopped, but sandbox termination could not be confirmed"
            )
        await asyncio.sleep(0.1)
