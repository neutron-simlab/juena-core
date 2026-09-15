"""Stub for 01/CP3. New in core — the identity seam (00-BOUNDARY.md,
decision 9). ``Principal`` is ``AuthenticatedUser`` with ``saml_subject``
renamed; core touches only ``.id``.

``session_principal`` is the cookie-reading dependency, formerly
``get_current_user``. ``SESSION_COOKIE_NAME`` stays ``"juena_session"`` — a
live cookie name; renaming it logs everyone out.

``local_principal`` is v2's entire identity story for this phase (decision
9 part 3, decision 17): a dependency returning a ``Principal`` built from a
stable, configured UUID. Two things it must do that a real provider does for
free, both landing **with** the provider, not later:

1. Its publication guard runs at **startup**, inside ``create_app`` — not as
   a per-request dependency, which would fail the first call rather than
   refuse to boot.
2. It **upserts its ``users`` row** inside the database lifespan, after
   ``database_lifespan`` has an engine and before the app serves — otherwise
   the first ``Chat`` insert hits an unresolved foreign key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

__all__ = [
    "SESSION_COOKIE_NAME",
    "Principal",
    "hash_session_token",
    "upsert_principal",
    "create_auth_session",
    "revoke_auth_session",
    "revoke_sessions_for_subject",
    "session_principal",
    "local_principal",
]

SESSION_COOKIE_NAME = "juena_session"


@dataclass(frozen=True, slots=True)
class Principal:
    """Stub — implemented in 01/CP3."""

    id: UUID
    subject: str
    issuer: str
    email: str | None
    display_name: str | None


def hash_session_token(token: str) -> str:
    raise NotImplementedError("juena_core.server.identity.hash_session_token lands in 01/CP3")


async def upsert_principal(*args: Any, **kwargs: Any) -> Principal:
    raise NotImplementedError("juena_core.server.identity.upsert_principal lands in 01/CP3")


async def create_auth_session(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.server.identity.create_auth_session lands in 01/CP3")


async def revoke_auth_session(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("juena_core.server.identity.revoke_auth_session lands in 01/CP3")


async def revoke_sessions_for_subject(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("juena_core.server.identity.revoke_sessions_for_subject lands in 01/CP3")


async def session_principal(*args: Any, **kwargs: Any) -> Principal:
    raise NotImplementedError("juena_core.server.identity.session_principal lands in 01/CP3")


def local_principal(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("juena_core.server.identity.local_principal lands in 01/CP3")
