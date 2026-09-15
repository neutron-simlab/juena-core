"""Stub for 01/CP5. Ported from ``juena-chatbot/app/client_setup.py`` — but
**not verbatim**, unlike what 00-BOUNDARY.md, decision 7 assumed.

**Finding, made during 01/CP0's stubbing pass, not by decision 7's own
measurement.** Decision 7 counted this module among six that "import
exactly six juena modules, all of which are core" and moved it whole on that
basis. But the source module also *defines*
``JUENA_AGENT_ID = "juena"`` and uses it as ``initialize_client``'s default
``agent_id`` — an application literal, not merely an application import.
juena-chatbot's own comment on the constant explains why: the UI has to name
an agent id before a session exists to ask the server which one is default.

Decision 7's import-surface measurement is real and still correct about the
*imports*; it did not catch this literal because it wasn't looking for one.
**Correction applied here:** ``initialize_client`` takes ``agent_id`` with no
default — each application's own UI module supplies its own id (v2 passes
``"simulator"`` or ``"advanced_mode"``; juena-chatbot's own thin wrapper
keeps passing ``"juena"``). Also typed against ``BaseAgentClient``, not the
source's ``AgentClient``.
"""

from __future__ import annotations

from juena_core.clients.base import BaseAgentClient

__all__ = ["initialize_client"]


def initialize_client(
    internal_api_url: str,
    agent_id: str,
    session_token: str | None = None,
) -> BaseAgentClient:
    raise NotImplementedError("juena_core.ui.client_setup.initialize_client lands in 01/CP5")
