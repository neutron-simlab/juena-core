"""Postgres models for identity, sessions, and chat ownership.

Core owns ``Base``, ``User``, ``AuthSession`` and ``Chat``. An application
adds its own tables by importing ``Base`` from here, which also means a model
class nobody imports is never created by ``create_all`` — the application
imports its models module for that side effect.

``User.issuer`` and ``User.subject`` are Python names for the columns
``idp_entity_id`` and ``saml_subject``. Do not rename the columns: the schema is
created by ``Base.metadata.create_all``, which never renames a column, so the
populated old column would sit beside a new empty one.
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
    """Return the current time in UTC.

    Returns:
        A timezone-aware datetime.
    """

    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative model base shared by core and its applications."""


class User(Base):
    """One person who has signed in, or the fixed local principal.

    Attributes:
        id: The user's id, used as ``Principal.id``.
        issuer: Opaque identity-provider entity id: a SAML issuer, or a fixed
            local value when the application has no identity provider. Stored
            in the ``idp_entity_id`` column.
        subject: Opaque subject within that issuer, such as a SAML NameID.
            Stored in the ``saml_subject`` column.
        email: The email address, if known.
        display_name: The name to show, if known.
        is_active: The sessions of an inactive user are refused.
        created_at: When the row was created.
        updated_at: When the row last changed.
        last_login_at: When the user last signed in.
        sessions: The user's sign-in sessions.
        chats: The user's conversations.
    """

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("idp_entity_id", "saml_subject", name="uq_users_idp_subject"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    issuer: Mapped[str] = mapped_column("idp_entity_id", Text, nullable=False)
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
    """One sign-in session, stored by the hash of its cookie token.

    Attributes:
        token_hash: SHA-256 hex digest of the session token; the token itself
            is never stored.
        user_id: The id of the user the session belongs to.
        created_at: When the session was issued.
        expires_at: When the session stops working.
        last_seen_at: When the session last authenticated a request.
        revoked_at: When the session was signed out, if it was.
        user: The user the session belongs to.
    """

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

    Attributes:
        thread_id: The conversation's thread id, which keys its checkpoints.
        user_id: The owner's id.
        agent_id: Which registered graph owns this conversation.
        title: The title shown in the chat list.
        summary: A summary of the conversation, set by the client.
        created_at: When the chat was created.
        updated_at: When the chat last changed; chat lists sort by it.
        user: The owner.
    """

    __tablename__ = "chats"
    __table_args__ = (Index("ix_chats_user_updated", "user_id", "updated_at"),)

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    #: Not nullable: a thread resumed under the wrong graph would get a
    #: checkpoint that does not fit that graph's state.
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
