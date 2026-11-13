"""Client construction for the Streamlit UI.

``agent_id`` and ``timeout`` are required, with no default:

- ``agent_id``, because a shared package must not carry one application's
  agent id into another.
- ``timeout``, because reading it from ``settings()`` would make
  ``juena_core.configure()`` a hidden precondition of building a client. A
  Streamlit process otherwise needs no core configuration, so it would fail at
  the first page load without saying why.

The UI talks to the API over HTTP and must not import the server package to
learn an agent id: the two run as separate processes, and in production as
separate containers.

An application that adds routes passes its ``BaseAgentClient`` subclass as
``client_class``.
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

    Returns:
        A client of ``client_class``.
    """
    return client_class(
        base_url=internal_api_url,
        agent=agent_id,
        session_token=session_token,
        timeout=timeout,
    )
