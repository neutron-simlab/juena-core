"""Client construction for the Streamlit UI.

Ported from ``juena-chatbot/app/client_setup.py`` — but **not verbatim**,
unlike what 00-BOUNDARY.md, decision 7 assumed.

**Finding, made during 01/CP0's stubbing pass rather than by decision 7's own
measurement.** Decision 7 counted this module among six that "import exactly
six juena modules, all of which are core" and moved it whole on that basis.
The source module also *defines* ``JUENA_AGENT_ID = "juena"`` and uses it as
``initialize_client``'s default ``agent_id`` — an application literal, not
merely an application import. Decision 7's import-surface measurement is real
and still correct about the imports; it was not looking for a literal.

**Two corrections, the same in kind.** ``agent_id`` and ``timeout`` are both
required, with no default:

- ``agent_id`` because a shared package must not smuggle one application's
  identity into the other. juena-chatbot's own thin wrapper keeps its
  ``JUENA_AGENT_ID``; v2 passes ``"simulator"`` or ``"advanced_mode"``.
- ``timeout`` because the source read it from ``global_config``, and core's
  equivalent would be ``settings()``. A Streamlit process otherwise needs no
  core configuration at all, so reading it here would make
  ``juena_core.configure()`` a hidden precondition of building a client —
  failing at the first page load, in the process least able to explain why.

The UI talks to the API over HTTP and must not import the server package to
learn an agent id: the two run as separate processes, and in production as
separate containers.

An application that adds routes passes its ``BaseAgentClient`` subclass through
``client_class``. The factory keeps the common constructor in one place without
teaching core which extra endpoints that subclass owns.
"""

from __future__ import annotations

from typing import TypeVar

from juena_core.clients.base import BaseAgentClient

__all__ = ["initialize_client"]

ClientT = TypeVar("ClientT", bound=BaseAgentClient)


def initialize_client(
    internal_api_url: str,
    agent_id: str,
    session_token: str | None = None,
    *,
    timeout: float,
    client_class: type[ClientT] = BaseAgentClient,
) -> ClientT:
    """Build a client for one agent against the private API URL.

    Args:
        internal_api_url: Base URL of the API, reachable from the UI process.
        agent_id: The registered agent this client streams to.
        session_token: The signed-in user's session cookie, when there is one.
        timeout: Request timeout in seconds. Streaming reads are exempt.
        client_class: The application's client subclass when it adds routes of
            its own. It must retain ``BaseAgentClient``'s constructor contract.
    """
    return client_class(
        base_url=internal_api_url,
        agent=agent_id,
        session_token=session_token,
        timeout=timeout,
    )
