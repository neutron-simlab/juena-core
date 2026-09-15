"""Stub for 01/CP2. Ported from ``juena/agents/specialist_outcome.py``, plus
two new middlewares (00-BOUNDARY.md, decision 15).

**Edit 2 of four.** ``_executions(state)`` reads ``state["execution_events"]``
and validates each entry against ``schema.interrupts.ExecutionEvidence``,
instead of importing ``juena.sandbox.artifacts.get_artifact_store``.
``_identity(runtime)`` uses ``runtime_context._context_value``. ``execution_events``
is a plain state channel — juena-chatbot's sandbox backend writes it, v2's
MCP wrapper writes it. One channel name, two writers; no protocol class, no
injected reader.

**Edit 4 of four — the root execution-evidence pair.**
``SpecialistOutcomeMiddleware`` runs inside specialists only
(installed in ``build_specialist_middleware``). v2's ``run_simulation`` is
called by the root supervisor, so a second middleware composes the
``<verified_by_server>`` block for the root response: factor the shared
composition logic out of this module rather than writing it twice.
``ArtifactMessageMiddleware`` moves into core alongside it — it was already
generic, and lived in juena's sandbox package only by accident of history.

``execution_events`` is declared with a list reducer, because several
modules in one simulation pipeline each append, and ``PrivateStateAttr`` so
it never crosses a delegation boundary as ordinary state. **A list reducer on
a checkpointed root graph accumulates forever** — each entry must carry the
``graph_run_id`` it belongs to (00-BOUNDARY.md, decision 16), and
``ExecutionEvidenceMiddleware`` must report only entries from the current
invocation, or turn five's answer re-reports turns one through four as fresh
evidence inside the one block that must never overstate what happened.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage

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
OUTCOME_VERIFIED_KEY = "juena_outcome_verified"
NO_REPORT = ""
NO_SUCCESSFUL_RUN = ""
UNVERIFIED_DIRECTIVE = ""


class SpecialistOutcomeState(AgentState):
    """Stub — implemented in 01/CP2. Declares ``execution_events`` with a
    list reducer and ``PrivateStateAttr``, keyed to ``graph_run_id`` per
    entry (00-BOUNDARY.md, decision 16)."""

    execution_events: NotRequired[Annotated[list[ExecutionEvidence], operator.add, PrivateStateAttr]]


def outcome_verified(message: AIMessage) -> bool:
    raise NotImplementedError("juena_core.agents.specialist_outcome.outcome_verified lands in 01/CP2")


def _identity(runtime: Any) -> tuple[str, str] | None:
    raise NotImplementedError("juena_core.agents.specialist_outcome._identity lands in 01/CP2")


def _executions(state: dict[str, Any]) -> list[ExecutionEvidence]:
    raise NotImplementedError("juena_core.agents.specialist_outcome._executions lands in 01/CP2")


def _report_block(report: SpecialistReport | None) -> str:
    raise NotImplementedError("juena_core.agents.specialist_outcome._report_block lands in 01/CP2")


def _execution_summary(executions: list[ExecutionEvidence]) -> str:
    raise NotImplementedError("juena_core.agents.specialist_outcome._execution_summary lands in 01/CP2")


def _artifact_summary(artifacts: list[ResultArtifactEvidence]) -> str:
    raise NotImplementedError("juena_core.agents.specialist_outcome._artifact_summary lands in 01/CP2")


def _verified_block(*args: Any, **kwargs: Any) -> str:
    raise NotImplementedError("juena_core.agents.specialist_outcome._verified_block lands in 01/CP2")


class SpecialistOutcomeMiddleware(AgentMiddleware[SpecialistOutcomeState]):
    """Stub — implemented in 01/CP2. Installed inside
    ``build_specialist_middleware`` only."""


class ExecutionEvidenceMiddleware(AgentMiddleware[SpecialistOutcomeState]):
    """Stub — implemented in 01/CP2 (00-BOUNDARY.md, decision 15). Spliced
    into ``build_supervisor_middleware(extra=...)`` for the root supervisor."""


class ArtifactMessageMiddleware(AgentMiddleware):
    """Stub — implemented in 01/CP2. Moved in from ``juena.sandbox``
    (00-BOUNDARY.md, decision 15); attaches artifact references to the
    outgoing message."""
