"""Server-authored specialist and root execution-evidence messages."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, merge_content

from juena_core.artifacts import (
    ARTIFACT_MESSAGE_KEY,
    get_artifact_store,
)
from juena_core.runtime_context import _context_value
from juena_core.schema.agents import ResultArtifactEvidence, SpecialistReport
from juena_core.schema.interrupts import ExecutionEvidence

__all__ = [
    "REPORT_OPEN",
    "REPORT_CLOSE",
    "VERIFIED_OPEN",
    "VERIFIED_CLOSE",
    "OUTCOME_VERIFIED_KEY",
    "NO_REPORT",
    "NO_SUCCESSFUL_RUN",
    "UNVERIFIED_DIRECTIVE",
    "SpecialistOutcomeState",
    "outcome_verified",
    "SpecialistOutcomeMiddleware",
    "ExecutionEvidenceMiddleware",
    "ArtifactMessageMiddleware",
]

REPORT_OPEN = "<specialist_report>"
REPORT_CLOSE = "</specialist_report>"
VERIFIED_OPEN = "<verified_by_server>"
VERIFIED_CLOSE = "</verified_by_server>"

# Preserved because detached callers read this key from checkpointed messages.
OUTCOME_VERIFIED_KEY = "juena_outcome_verified"

NO_REPORT = (
    "The specialist returned no structured report. Its run ended early -- most "
    "often because the model-call budget was exhausted -- so nothing it said "
    "mid-run can be treated as a result."
)
NO_SUCCESSFUL_RUN = (
    "Every execution in this run failed. No calculation, fit, or plot was produced."
)
UNVERIFIED_DIRECTIVE = (
    "Tell the user this attempt failed and say why. Use ask_user to offer concrete "
    "next steps. Do not report results, values, plots, or file names -- none exist."
)


class SpecialistOutcomeState(AgentState):
    """Private execution history and per-specialist artifact baseline."""

    specialist_initial_artifact_ids: NotRequired[
        Annotated[list[str], PrivateStateAttr]
    ]
    execution_events: NotRequired[
        Annotated[list[ExecutionEvidence], operator.add, PrivateStateAttr]
    ]


def outcome_verified(message: AIMessage) -> bool:
    """Return whether the server marked a specialist hand-off as verified."""

    return message.additional_kwargs.get(OUTCOME_VERIFIED_KEY) is True


def _identity(runtime: Any) -> tuple[str, str] | None:
    context = getattr(runtime, "context", None)
    user_id = _context_value(context, "user_id")
    thread_id = _context_value(context, "thread_id")
    if not user_id or not thread_id:
        return None
    return user_id, thread_id


def _graph_run_id(runtime: Any) -> str | None:
    """The invocation this evidence belongs to, from the context first.

    Same order, and the same reason, as
    :func:`juena_core.sandbox.middleware._graph_run_id`: evidence is written
    inside a subagent and read outside it, and only the runtime context holds
    one value on both sides of that boundary.
    """

    value = _context_value(getattr(runtime, "context", None), "run_id")
    if value:
        return value
    execution_info = getattr(runtime, "execution_info", None)
    value = getattr(execution_info, "run_id", None)
    if value:
        return str(value)
    config = getattr(runtime, "config", None)
    value = config.get("run_id") if hasattr(config, "get") else None
    return str(value) if value else None


def _executions(
    state: Mapping[str, Any],
    *,
    graph_run_id: str | None = None,
) -> list[ExecutionEvidence]:
    """Validate evidence and optionally keep only one graph invocation."""

    events: list[ExecutionEvidence] = []
    for value in state.get("execution_events", []):
        try:
            event = ExecutionEvidence.model_validate(value)
        except ValueError:
            continue
        if graph_run_id is None or event.graph_run_id == graph_run_id:
            events.append(event)
    return events


def _report_block(report: SpecialistReport | None) -> str:
    if report is None:
        return f"{REPORT_OPEN}\nNone returned.\n{REPORT_CLOSE}"
    lines = [f"Status: {report.status}", "", f"Finding: {report.finding}"]
    for label, items in (
        ("Evidence", report.evidence),
        ("Actions", report.actions),
        ("Limitations", report.limitations),
    ):
        if items:
            lines.extend(["", f"{label}:", *[f"- {item}" for item in items]])
    body = "\n".join(lines)
    return f"{REPORT_OPEN}\n{body}\n{REPORT_CLOSE}"


def _execution_summary(executions: list[ExecutionEvidence]) -> str:
    if not executions:
        return "Executions: none."
    detail = ", ".join(
        f"{event.status}"
        + ("" if event.exit_code is None else f" (exit {event.exit_code})")
        for event in executions
    )
    return f"Executions: {len(executions)} -- {detail}."


def _artifact_summary(artifacts: list[ResultArtifactEvidence]) -> str:
    if not artifacts:
        return "Result artifacts delivered: none."
    detail = "; ".join(
        f"{item.filename} ({item.kind}, id {item.artifact_id})" for item in artifacts
    )
    return f"Result artifacts delivered: {detail}."


def _verified_block(
    *,
    subject_label: str,
    subject: str,
    executions: list[ExecutionEvidence],
    artifacts: list[ResultArtifactEvidence],
    undelivered: list[tuple[str, str]],
    failures: list[str],
) -> str:
    lines = [f"{subject_label}: {subject}"]
    if failures:
        lines.append("STATUS: UNVERIFIED")
        lines.extend(f"- {reason}" for reason in failures)
    else:
        lines.append("STATUS: VERIFIED")
    lines.append(_execution_summary(executions))
    lines.append(_artifact_summary(artifacts))
    if undelivered:
        lines.append("Produced but NOT delivered to the user:")
        lines.extend(f"- {filename}: {reason}" for filename, reason in undelivered)
    if failures:
        lines.append(UNVERIFIED_DIRECTIVE)
    body = "\n".join(lines)
    return f"{VERIFIED_OPEN}\n{body}\n{VERIFIED_CLOSE}"


def _execution_failures(executions: list[ExecutionEvidence]) -> list[str]:
    attempted = [event for event in executions if event.attempted]
    return [NO_SUCCESSFUL_RUN] if attempted and not any(
        event.succeeded for event in attempted
    ) else []


def _final_ai_message(state: Any) -> AIMessage | None:
    messages = state.get("messages", []) if isinstance(state, Mapping) else []
    last_ai = next(
        (item for item in reversed(messages) if isinstance(item, AIMessage)),
        None,
    )
    return None if last_ai is None or last_ai.tool_calls else last_ai


class SpecialistOutcomeMiddleware(AgentMiddleware[SpecialistOutcomeState]):
    """Pair a specialist report with facts the specialist cannot author."""

    state_schema = SpecialistOutcomeState

    def __init__(self, *, specialist_name: str) -> None:
        self._specialist_name = specialist_name

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        identity = _identity(runtime)
        if identity is None:
            return {"specialist_initial_artifact_ids": []}
        refs = get_artifact_store().peek_result_refs(*identity)
        return {"specialist_initial_artifact_ids": [ref.artifact_id for ref in refs]}

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        return self.before_agent(state, runtime)

    def _own_artifacts(
        self,
        state: Mapping[str, Any],
        identity: tuple[str, str] | None,
    ) -> tuple[list[ResultArtifactEvidence], list[tuple[str, str]]]:
        if identity is None:
            return [], []
        baseline = set(state.get("specialist_initial_artifact_ids", []))
        store = get_artifact_store()
        artifacts = [
            ResultArtifactEvidence(
                artifact_id=ref.artifact_id,
                filename=ref.filename,
                kind=ref.kind,
                mime_type=ref.mime_type,
                caption=ref.caption,
            )
            for ref in store.peek_result_refs(*identity)
            if ref.artifact_id not in baseline
        ]
        return artifacts, store.peek_undelivered(*identity)

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        values = state if isinstance(state, Mapping) else {}
        try:
            report = SpecialistReport.model_validate(values.get("structured_response"))
        except ValueError:
            report = None

        executions = _executions(values)
        artifacts, undelivered = self._own_artifacts(values, _identity(runtime))
        failures = ([] if report is not None else [NO_REPORT]) + _execution_failures(
            executions
        )

        if report is not None and undelivered:
            report = report.model_copy(
                update={
                    "limitations": [
                        *report.limitations,
                        *[
                            f"{name} was not delivered: {why}"
                            for name, why in undelivered
                        ],
                    ]
                }
            )

        package = "\n\n".join(
            [
                _report_block(report),
                _verified_block(
                    subject_label="Specialist",
                    subject=self._specialist_name,
                    executions=executions,
                    artifacts=artifacts,
                    undelivered=undelivered,
                    failures=failures,
                ),
            ]
        )
        return {
            "structured_response": None,
            "messages": [
                AIMessage(
                    content=package,
                    additional_kwargs={OUTCOME_VERIFIED_KEY: not failures},
                )
            ],
        }

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        return self.after_agent(state, runtime)


class ExecutionEvidenceMiddleware(AgentMiddleware[SpecialistOutcomeState]):
    """Append current-invocation execution facts to a root agent response."""

    state_schema = SpecialistOutcomeState

    def __init__(self, *, agent_name: str = "supervisor") -> None:
        self._agent_name = agent_name

    @staticmethod
    def _artifacts(
        executions: list[ExecutionEvidence],
        identity: tuple[str, str] | None,
    ) -> list[ResultArtifactEvidence]:
        if identity is None:
            return []
        user_id, _thread_id = identity
        store = get_artifact_store()
        artifacts: list[ResultArtifactEvidence] = []
        seen: set[str] = set()
        for event in executions:
            for artifact_id in event.artifact_ids:
                if artifact_id in seen:
                    continue
                record = store.get(user_id, artifact_id)
                if record is None:
                    continue
                ref, _content = record
                seen.add(artifact_id)
                artifacts.append(
                    ResultArtifactEvidence(
                        artifact_id=ref.artifact_id,
                        filename=ref.filename,
                        kind=ref.kind,
                        mime_type=ref.mime_type,
                        caption=ref.caption,
                    )
                )
        return artifacts

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last_ai = _final_ai_message(state)
        graph_run_id = _graph_run_id(runtime)
        if last_ai is None or graph_run_id is None:
            return None
        values = state if isinstance(state, Mapping) else {}
        executions = _executions(values, graph_run_id=graph_run_id)
        if not executions:
            return None
        artifacts = self._artifacts(executions, _identity(runtime))
        undelivered = [item for event in executions for item in event.dropped]
        block = _verified_block(
            subject_label="Agent",
            subject=self._agent_name,
            executions=executions,
            artifacts=artifacts,
            undelivered=undelivered,
            failures=_execution_failures(executions),
        )
        updated = last_ai.model_copy(
            update={"content": merge_content(last_ai.content, f"\n\n{block}")}
        )
        return {"messages": [updated]}

    async def aafter_agent(
        self,
        state: Any,
        runtime: Any,
    ) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)


class ArtifactMessageMiddleware(AgentMiddleware):
    """Attach pending artifacts to the root agent's final response."""

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last_ai = _final_ai_message(state)
        if last_ai is None:
            return None
        identity = _identity(runtime)
        if identity is None:
            return None
        artifacts = get_artifact_store().claim_for_message(*identity)
        if not artifacts:
            return None
        updated = last_ai.model_copy(
            update={
                "additional_kwargs": {
                    **last_ai.additional_kwargs,
                    ARTIFACT_MESSAGE_KEY: artifacts,
                }
            }
        )
        return {"messages": [updated]}

    async def aafter_agent(
        self,
        state: Any,
        runtime: Any,
    ) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)
