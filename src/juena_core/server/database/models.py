"""Stub for 01/CP3. Core's table ownership (00-BOUNDARY.md, decision 9 part
4): ``Base``, ``utc_now``, ``User``, ``AuthSession``, ``Chat``.
juena-chatbot's own models module holds ``SamlLoginRequest``, ``SandboxJob``
and ``ResearchJob``, importing ``Base`` from here.

**No column is renamed on disk.** There is no Alembic; ``create_all`` never
renames, so ``User.subject``/``User.issuer`` map to the *existing* columns
by positional name:

    subject: Mapped[str] = mapped_column("saml_subject", Text, nullable=False)
    issuer:  Mapped[str] = mapped_column("idp_entity_id", Text, nullable=False)

**``Chat`` gains ``agent_id``** (decision 13), not nullable — v2 has two
agents, and a conversation that does not know its own graph is the bug being
prevented.

Each stub class below is ``__abstract__`` purely so it satisfies SQLAlchemy's
declarative-class requirements before CP3 gives it real columns — remove the
flag along with adding them; it is not part of the final shape.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import DeclarativeBase

__all__ = ["utc_now", "Base", "User", "AuthSession", "Chat"]


def utc_now() -> datetime:
    raise NotImplementedError("juena_core.server.database.models.utc_now lands in 01/CP3")


class Base(DeclarativeBase):
    """Stub — implemented in 01/CP3."""


class User(Base):
    """Stub — implemented in 01/CP3. ``subject``/``issuer`` map onto the
    existing ``saml_subject``/``idp_entity_id`` columns by positional name."""

    __abstract__ = True


class AuthSession(Base):
    """Stub — implemented in 01/CP3."""

    __abstract__ = True


class Chat(Base):
    """Stub — implemented in 01/CP3. Gains a non-nullable ``agent_id``
    column the source model does not have."""

    __abstract__ = True
