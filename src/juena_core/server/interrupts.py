"""The registry of interrupt kinds a paused graph can be resumed from.

A LangGraph interrupt pauses a run until a person answers it. Two questions
have to be answered about every pending interrupt, and only the code that
raised it knows: **what card does the user see**, and **what does their reply
mean to the graph**. So a kind is registered with both.

Core registers exactly one kind, ``clarification``, because ``ask_user`` is
core's. juena-chatbot registers its sandbox ``execute_approval``; VITESS v2
registers nothing at all, and its ``/resume`` therefore accepts only the
clarification arm.

**Why one registration rather than decision 5's ``register_interrupt_event(
kind, builder)``.** An event builder alone lets core *show* a card it cannot
*resume*: the user answers, ``/resume`` finds no way to turn the answer into a
``Command``, and the run stays paused with no error that names the cause. The
two halves have the same owner and the same lifetime, so they are registered
together.

The event builder doubles as the classifier: it returns ``None`` for an
interrupt that is not its kind. "Can I render this?" and "what does it look
like?" are the same question asked twice, and answering it once means a kind
cannot be recognised but unrenderable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from juena_core.artifacts import get_artifact_store
from juena_core.log import get_logger
from juena_core.schema.interrupts import CLARIFICATION_KIND, ClarificationResumeInput
from juena_core.server.streaming.events import clarification_required_event

logger = get_logger(__name__)

__all__ = [
    "ResumeError",
    "InterruptKind",
    "register_interrupt_kind",
    "registered_interrupt_kinds",
    "interrupt_event",
    "classify_interrupt",
    "get_pending_interrupt",
    "first_pending_interrupt",
    "build_resume_command",
]


class ResumeError(ValueError):
    """Raised for stale, malformed, or unresumable interrupt replies."""


@dataclass(frozen=True, slots=True)
class InterruptKind:
    """How one kind of pause is shown to a person and resumed afterwards."""

    kind: str
    #: Build the SSE payload for this interrupt, or return ``None`` when the
    #: interrupt is not of this kind.
    event: Callable[[Interrupt], dict[str, Any] | None]
    #: Turn a validated reply into the ``Command`` that resumes the graph.
    #: Called with keyword arguments ``interrupt``, ``user_id`` and ``payload``.
    resume: Callable[..., Command]


_interrupt_kinds: dict[str, InterruptKind] = {}


def register_interrupt_kind(
    kind: str,
    *,
    event: Callable[[Interrupt], dict[str, Any] | None],
    resume: Callable[..., Command],
) -> None:
    """Register how one interrupt kind is rendered and resumed.

    ``kind`` must match the ``kind`` discriminator on the matching arm of the
    application's resume union, because that is what ``/resume`` dispatches on.
    """

    _interrupt_kinds[kind] = InterruptKind(kind=kind, event=event, resume=resume)
    logger.info("Registered interrupt kind: %s", kind)


def registered_interrupt_kinds() -> list[str]:
    return list(_interrupt_kinds)


def interrupt_event(interrupt: Interrupt) -> dict[str, Any] | None:
    """Build the SSE payload for one pending interrupt, or None to skip it."""

    if not isinstance(interrupt.value, dict):
        return None
    for registered in _interrupt_kinds.values():
        payload = registered.event(interrupt)
        if payload is not None:
            return payload
    return None


def classify_interrupt(interrupt: Interrupt) -> str | None:
    """The registered kind of one interrupt, or None when nothing claims it."""

    if not isinstance(interrupt.value, dict):
        return None
    for registered in _interrupt_kinds.values():
        if registered.event(interrupt) is not None:
            return registered.kind
    return None


def _pending_interrupts(snapshot: Any) -> list[Interrupt]:
    return [
        interrupt
        for task in getattr(snapshot, "tasks", ())
        for interrupt in getattr(task, "interrupts", ())
        if isinstance(interrupt, Interrupt)
    ]


async def get_pending_interrupt(
    agent: CompiledStateGraph,
    config: dict[str, Any],
    interrupt_id: str,
) -> Interrupt:
    """Find one pending interrupt on this thread by id.

    Resolving by id is what makes a question raised inside a specialist subgraph
    resume that specialist exactly where it paused.
    """

    snapshot = await agent.aget_state(config=config)
    for interrupt in _pending_interrupts(snapshot):
        if interrupt.id == interrupt_id:
            return interrupt
    raise ResumeError("This reply is stale, already used, or does not belong to this thread")


async def first_pending_interrupt(
    agent: CompiledStateGraph,
    config: dict[str, Any],
) -> tuple[Interrupt, str] | None:
    """Return the first renderable pending interrupt, including after a reload."""

    snapshot = await agent.aget_state(config=config)
    for interrupt in _pending_interrupts(snapshot):
        kind = classify_interrupt(interrupt)
        if kind is not None:
            return interrupt, kind
    return None


async def build_resume_command(
    *,
    agent: CompiledStateGraph,
    config: dict[str, Any],
    user_id: str,
    payload: Any,
) -> Command:
    """Validate one pending interrupt reply and build its resume command."""

    kind = getattr(payload, "kind", None)
    registered = _interrupt_kinds.get(kind) if isinstance(kind, str) else None
    if registered is None:
        raise ResumeError(f"No interrupt kind is registered for {kind!r}")

    interrupt = await get_pending_interrupt(agent, config, payload.interrupt_id)
    if classify_interrupt(interrupt) != registered.kind:
        raise ResumeError(f"This interrupt is not a pending {registered.kind}")
    return registered.resume(interrupt=interrupt, user_id=user_id, payload=payload)


def _clarification_event(interrupt: Interrupt) -> dict[str, Any] | None:
    value = interrupt.value
    if not isinstance(value, dict) or value.get("kind") != CLARIFICATION_KIND:
        return None
    if not isinstance(value.get("question"), str):
        return None
    return clarification_required_event(interrupt.id, value)


def _clarification_resume(
    *,
    interrupt: Interrupt,
    user_id: str,
    payload: ClarificationResumeInput,
) -> Command:
    answer = payload.answer.strip()
    if not answer:
        raise ResumeError("An answer must not be empty")
    get_artifact_store().audit(
        {
            "event": "clarification_answered",
            "user_id": user_id,
            "thread_id": payload.thread_id,
            "interrupt_id": payload.interrupt_id,
        }
    )
    # The tool returns this string, so the asking agent resumes with the user's
    # own words rather than a decision code.
    return Command(resume={payload.interrupt_id: answer})


register_interrupt_kind(
    CLARIFICATION_KIND,
    event=_clarification_event,
    resume=_clarification_resume,
)
