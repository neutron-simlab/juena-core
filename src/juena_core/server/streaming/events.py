"""SSE payload builders for the agent stream.

One place that knows the wire shape, so the frame format is not spelled out as
``f"data: {json.dumps(...)}"`` at a dozen call sites.

Event types on the wire:

``thread``   thread id, emitted once before any content
``token``    an incremental text chunk of the assistant's answer
``message``  a complete message (`ChatMessage` payload)
``status``   agent activity: thinking / researching / running a tool
``error``    a failure the client should surface

Two more are raised by a paused graph and built through the interrupt-kind
registry in :mod:`juena_core.server.interrupts`: ``clarification_required``,
which core owns because ``ask_user`` is core's, and whatever kinds the
application registers.
"""

from __future__ import annotations

from typing import Any

from juena_core.schema.server import ChatMessage

__all__ = [
    "PHASE_THINKING",
    "PHASE_RESEARCHING",
    "PHASE_TOOL",
    "PHASE_DONE",
    "thread_event",
    "token_event",
    "message_event",
    "status_event",
    "error_event",
    "clarification_required_event",
    "approval_required_event",
]

# Phases a client can render. `researching` means a subagent is working.
PHASE_THINKING = "thinking"
PHASE_RESEARCHING = "researching"
PHASE_TOOL = "tool"
PHASE_DONE = "done"


def thread_event(thread_id: str) -> dict[str, Any]:
    return {"type": "thread", "thread_id": thread_id}


def token_event(content: str) -> dict[str, Any]:
    return {"type": "token", "content": content}


def message_event(message: ChatMessage) -> dict[str, Any]:
    return {"type": "message", "content": message.model_dump(mode="json")}


def status_event(phase: str, label: str, **extra: Any) -> dict[str, Any]:
    return {"type": "status", "phase": phase, "label": label, **extra}


def error_event(content: str) -> dict[str, Any]:
    return {"type": "error", "content": content}


def clarification_required_event(interrupt_id: str, value: dict[str, Any]) -> dict[str, Any]:
    """Surface a question an agent raised with ``ask_user``."""

    options = value.get("options")
    return {
        "type": "clarification_required",
        "interrupt_id": interrupt_id,
        "asked_by": value.get("asked_by") or "",
        "question": value.get("question") or "",
        "options": [item for item in options if isinstance(item, str)]
        if isinstance(options, list)
        else [],
    }


def approval_required_event(
    interrupt_id: str,
    value: dict[str, Any],
    *,
    extra_limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Surface a pending ``HumanInTheLoopMiddleware`` action for approval.

    The ``action_requests``/``review_configs`` shape is LangChain's, so the
    unpacking is shared. ``extra_limits`` is not: the card tells a person what
    the action is allowed to do, and only the application knows — "network:
    none, 2 vCPU, 4 GB" describes juena-chatbot's Podman sandbox and would be a
    false promise from anywhere else.
    """

    action_requests = value.get("action_requests") or []
    review_configs = value.get("review_configs") or []
    action = action_requests[0] if action_requests and isinstance(action_requests[0], dict) else {}
    review = review_configs[0] if review_configs and isinstance(review_configs[0], dict) else {}
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    description = action.get("description")
    if isinstance(description, str) and "\n\nTool:" in description:
        description = description.split("\n\nTool:", 1)[0]
    return {
        "type": "approval_required",
        "interrupt_id": interrupt_id,
        "action_name": action.get("name"),
        "command": args.get("command"),
        "description": description,
        "limits": {"timeout_seconds": args.get("timeout"), **(extra_limits or {})},
        "allowed_decisions": review.get("allowed_decisions") or [],
    }
