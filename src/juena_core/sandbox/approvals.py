"""Human approval contract for sandbox ``execute`` calls."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from langgraph.types import Command, Interrupt
from pydantic import BaseModel, Field, model_validator

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


class ApprovalDecision(BaseModel):
    """One decision in the order of the pending commands."""

    decision: Literal["approve", "edit", "reject"]
    edited_command: str | None = Field(default=None, max_length=100_000)


class ApprovalResumeInput(ResumeBase):
    """Review one command or every command in a pending approval batch."""

    kind: Literal["execute_approval"] = "execute_approval"
    decision: Literal["approve", "edit", "reject"] | None = None
    edited_command: str | None = Field(default=None, max_length=100_000)
    decisions: list[ApprovalDecision] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def one_decision_format(self) -> Self:
        if (self.decision is None) == (self.decisions is None):
            raise ValueError("Provide either decision or decisions")
        if self.decisions is not None and self.edited_command is not None:
            raise ValueError("Put edited_command inside its batch decision")
        return self


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


def _execute_actions(interrupt: Interrupt) -> list[dict[str, Any]]:
    value = interrupt.value
    if not isinstance(value, dict):
        raise ResumeError("Malformed sandbox approval request")
    actions = value.get("action_requests")
    if not isinstance(actions, list) or not actions:
        raise ResumeError("Expected sandbox actions")
    for action in actions:
        if not isinstance(action, dict):
            raise ResumeError("Malformed sandbox action")
        args = action.get("args")
        if action.get("name") != "execute" or not isinstance(args, dict):
            raise ResumeError("Only sandbox execute approvals are accepted")
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ResumeError("Sandbox command is missing")
    return actions


def _approval_event(interrupt: Interrupt) -> dict[str, Any] | None:
    try:
        _execute_actions(interrupt)
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
    actions = _execute_actions(interrupt)
    choices = payload.decisions
    if choices is None:
        choices = [ApprovalDecision(decision=payload.decision, edited_command=payload.edited_command)]
    if len(choices) != len(actions):
        raise ResumeError("Provide a decision for every pending sandbox command")
    decisions = [
        _command_decision(action, choice)
        for action, choice in zip(actions, choices)
    ]

    for index, (action, choice) in enumerate(zip(actions, choices)):
        get_artifact_store().audit(
            {
                "event": "sandbox_approval",
                "user_id": user_id,
                "thread_id": payload.thread_id,
                "interrupt_id": payload.interrupt_id,
                "action_index": index,
                "original_command": action["args"]["command"],
                "edited_command": choice.edited_command.strip() if choice.decision == "edit" else None,
                "decision": choice.decision,
            }
        )
    return Command(resume={payload.interrupt_id: {"decisions": decisions}})


def _command_decision(action: dict[str, Any], choice: ApprovalDecision) -> dict[str, Any]:
    if choice.decision == "approve":
        if choice.edited_command is not None:
            raise ResumeError("edited_command is valid only for an edit decision")
        return {"type": "approve"}
    if choice.decision == "reject":
        if choice.edited_command is not None:
            raise ResumeError("edited_command is valid only for an edit decision")
        return {
            "type": "reject",
            "message": "The user rejected this sandbox command.",
        }
    edited_command = (choice.edited_command or "").strip()
    if not edited_command:
        raise ResumeError("An edited command must not be empty")
    return {
        "type": "edit",
        "edited_action": {
            "name": "execute",
            "args": {**action["args"], "command": edited_command},
        },
    }


def register_sandbox_interrupt() -> None:
    """Register rendering and resume behavior for sandbox approvals."""

    register_interrupt_kind(
        EXECUTE_APPROVAL_KIND,
        event=_approval_event,
        resume=_execute_resume,
    )
