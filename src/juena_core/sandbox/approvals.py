"""Human approval contract for sandbox ``execute`` calls."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langgraph.types import Command, Interrupt
from pydantic import Field

from juena_core.artifacts import get_artifact_store
from juena_core.sandbox.config import sandbox_settings
from juena_core.sandbox.policy import requires_execution_approval
from juena_core.schema.interrupts import ClarificationResumeInput, ResumeBase
from juena_core.server.interrupts import ResumeError, register_interrupt_kind
from juena_core.server.streaming.events import approval_required_event

__all__ = [
    "EXECUTE_APPROVAL_KIND",
    "ApprovalResumeInput",
    "SandboxResumeInput",
    "sandbox_interrupt_on",
    "register_sandbox_interrupt",
]

EXECUTE_APPROVAL_KIND = "execute_approval"


class ApprovalResumeInput(ResumeBase):
    """Approve, edit, or reject one pending sandbox command."""

    kind: Literal["execute_approval"] = "execute_approval"
    decision: Literal["approve", "edit", "reject"]
    edited_command: str | None = Field(default=None, max_length=100_000)


SandboxResumeInput = Annotated[
    ApprovalResumeInput | ClarificationResumeInput,
    Field(discriminator="kind"),
]


def sandbox_interrupt_on(
    description: str = "Review the generated command before it runs in the isolated sandbox.",
) -> dict[str, Any]:
    """Return the ``HumanInTheLoopMiddleware`` policy for ``execute``."""

    return {
        "execute": {
            "allowed_decisions": ["approve", "edit", "reject"],
            "description": description,
            "when": requires_execution_approval,
        }
    }


def _execute_action(interrupt: Interrupt) -> dict[str, Any]:
    value = interrupt.value
    if not isinstance(value, dict):
        raise ResumeError("Malformed sandbox approval request")
    actions = value.get("action_requests")
    if (
        not isinstance(actions, list)
        or len(actions) != 1
        or not isinstance(actions[0], dict)
    ):
        raise ResumeError("Expected exactly one sandbox action")
    action = actions[0]
    args = action.get("args")
    if action.get("name") != "execute" or not isinstance(args, dict):
        raise ResumeError("Only sandbox execute approvals are accepted")
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ResumeError("Sandbox command is missing")
    return action


def _approval_event(interrupt: Interrupt) -> dict[str, Any] | None:
    try:
        _execute_action(interrupt)
    except ResumeError:
        return None
    config = sandbox_settings()
    value = interrupt.value
    assert isinstance(value, dict)
    return approval_required_event(
        interrupt.id,
        value,
        extra_limits={
            "network": "none",
            "cpu": config.cpu_limit,
            "memory": config.memory_limit,
        },
    )


def _execute_resume(
    *,
    interrupt: Interrupt,
    user_id: str,
    payload: ApprovalResumeInput,
) -> Command:
    action = _execute_action(interrupt)
    original_args = dict(action["args"])

    if payload.decision == "approve":
        if payload.edited_command is not None:
            raise ResumeError("edited_command is valid only for an edit decision")
        decision: dict[str, Any] = {"type": "approve"}
        edited_command = None
    elif payload.decision == "reject":
        if payload.edited_command is not None:
            raise ResumeError("edited_command is valid only for an edit decision")
        decision = {
            "type": "reject",
            "message": "The user rejected this sandbox command.",
        }
        edited_command = None
    else:
        edited_command = (payload.edited_command or "").strip()
        if not edited_command:
            raise ResumeError("An edited command must not be empty")
        decision = {
            "type": "edit",
            "edited_action": {
                "name": "execute",
                "args": {**original_args, "command": edited_command},
            },
        }

    get_artifact_store().audit(
        {
            "event": "sandbox_approval",
            "user_id": user_id,
            "thread_id": payload.thread_id,
            "interrupt_id": payload.interrupt_id,
            "original_command": original_args["command"],
            "edited_command": edited_command,
            "decision": payload.decision,
        }
    )
    return Command(resume={payload.interrupt_id: {"decisions": [decision]}})


def register_sandbox_interrupt() -> None:
    """Register rendering and resume behavior for sandbox approvals."""

    register_interrupt_kind(
        EXECUTE_APPROVAL_KIND,
        event=_approval_event,
        resume=_execute_resume,
    )
