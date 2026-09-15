"""Postgres models for identity, sessions, and chat ownership.

Core owns ``Base``, ``User``, ``AuthSession`` and ``Chat``. An application
adds its own tables by importing ``Base`` from here, which also means a model
class nobody imports is never created by ``create_all`` — the application
imports its models module for that side effect.

``User.subject`` and ``User.issuer`` are deliberately *attribute* renames over
the original column names. There is no Alembic here: the schema is created by
``Base.metadata.create_all``, which never renames a column, so renaming these
on disk would leave the populated ``saml_subject`` column sitting beside a new
empty one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = ["utc_now", "Base", "User", "AuthSession", "Chat"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative model base shared by core and its applications."""


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("idp_entity_id", "saml_subject", name="uq_users_idp_subject"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    #: Opaque identity-provider entity id. A SAML issuer in juena-chatbot, a
    #: fixed local value where the application ships no identity provider.
    issuer: Mapped[str] = mapped_column("idp_entity_id", Text, nullable=False)
    #: Opaque subject within that issuer, a SAML NameID in juena-chatbot.
    subject: Mapped[str] = mapped_column("saml_subject", Text, nullable=False)
    email: Mapped[str | None] = mapped_column(Text)
    display_name: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    sessions: Mapped[list["AuthSession"]] = relationship(back_populates="user")
    chats: Mapped[list["Chat"]] = relationship(back_populates="user")


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="sessions")


class Chat(Base):
    """Ownership and display metadata for one conversation thread.

    Message content deliberately lives only in the LangGraph checkpointer
    tables; mirroring it here would give the UI and the agent two histories
    that can drift apart.
    """

    __tablename__ = "chats"
    __table_args__ = (Index("ix_chats_user_updated", "user_id", "updated_at"),)

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    #: Which registered graph owns this conversation. Not nullable: an
    #: application with two agents resumes a thread under the graph named in
    #: the request, and a checkpoint whose state channels belong to the other
    #: graph is the failure this column exists to prevent.
    agent_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="New Chat")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    user: Mapped[User] = relationship(back_populates="chats")
