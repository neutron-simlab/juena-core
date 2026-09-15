"""Stub for 01/CP4. New in core, assembling what ``juena/server/service.py``
does today into a factory (00-BOUNDARY.md, decision 9 part 2).

**Two side-effect-import traps live here**, both transplanted rather than
re-derived: the agent-registry one (a factory not imported is never
registered, so the process serving `uvicorn juena_core...:app` gets an empty
registry) and the database-model one from 01/CP3 (a model class not imported
is never created by ``create_all``, and it fails at first write, not at
startup). Each application imports its own agent module and its own models
module with a ``# noqa: F401`` and a comment, in its own ``service.py``.

``register_interrupt_event`` replaces ``processor.py``'s branch on
``interrupt_kind(...) == EXECUTE_APPROVAL_KIND`` (00-BOUNDARY.md, decision
5): core registers the clarification kind it knows; the application
registers its own kinds at startup.

**Two things CP3 built for this factory to call** (``server/identity.py``),
both no-ops for a real identity provider:

- ``refuse_published_api(principal)`` — raises when a fixed principal, which
  authenticates nobody, is paired with a published API. ``local_principal``
  already calls it at construction, so this is the second of two gates rather
  than the only one; call it anyway, because an application may build its
  principal before it decides how to serve it.
- ``ensure_principal_row(fixed_principal(principal))`` — when
  ``fixed_principal`` returns a principal, its ``users`` row must be written
  inside the database lifespan, after the session factory exists and before
  the app serves. Nothing else creates it, and ``chats.user_id`` is a foreign
  key.

**Done when** (01/CP4): the route list this factory produces includes
``/chats``, ``/chats/{thread_id}``, ``/stream``, ``/stream_with_files``,
``/resume``, ``/{agent_id}/stream``, ``/{agent_id}/stream_with_files``,
``/{agent_id}/resume``, ``/threads/{thread_id}``,
``/threads/{thread_id}/pending-interrupt``, ``/artifacts/{artifact_id}``,
``/health`` — and nothing under ``/auth/``, because SAML is not here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from juena_core.server.identity import Principal

__all__ = ["register_interrupt_event", "create_app"]

_interrupt_event_builders: dict[str, Callable[..., dict[str, Any]]] = {}


def register_interrupt_event(kind: str, builder: Callable[..., dict[str, Any]]) -> None:
    raise NotImplementedError("juena_core.server.service.register_interrupt_event lands in 01/CP4")


def create_app(
    *,
    principal: Callable[..., Principal],
    extra_routers: tuple[Any, ...] = (),
    extra_lifespans: tuple[Any, ...] = (),
) -> Any:
    raise NotImplementedError("juena_core.server.service.create_app lands in 01/CP4")
