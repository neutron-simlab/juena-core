"""The identity seam: who a request belongs to, and who owns a thread.

Core defines the *shape* of an authenticated caller and the session storage
behind it. It does not define how anyone authenticates. An application passes
a dependency that returns a :class:`Principal`; juena-chatbot passes one
backed by SAML, and an application with no identity provider passes
:func:`local_principal`.

Core's own code touches only ``Principal.id``, which is what reaches
``RuntimeModelContext.user_id`` and every ``user_id`` foreign key.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from juena_core.config import settings
from juena_core.log import get_logger
from juena_core.server.database.connection import get_db_session, get_session_factory
from juena_core.server.database.models import AuthSession, User, utc_now

logger = get_logger(__name__)

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
    "fixed_principal",
    "refuse_published_api",
    "ensure_principal_row",
]

#: A live cookie name. Renaming it signs every existing session out.
SESSION_COOKIE_NAME = "juena_session"

#: Set by :func:`local_principal` on the dependency it returns, carrying the
#: principal itself. One attribute answers both questions the server asks:
#: whether this dependency authenticates anybody, and which ``users`` row has
#: to exist before it can own a conversation.
_FIXED_PRINCIPAL_ATTR = "__juena_core_fixed_principal__"

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


@dataclass(frozen=True, slots=True)
class Principal:
    """One authenticated caller, however the application authenticated them."""

    id: UUID
    #: Opaque within ``issuer``; a SAML NameID in juena-chatbot.
    subject: str
    #: Opaque identity-provider id; a SAML entity id in juena-chatbot.
    issuer: str
    email: str | None
    display_name: str | None


#: What ``create_app`` accepts as its identity provider. FastAPI resolves it as
#: a dependency, so it may be a plain function or a coroutine function.
PrincipalDependency = Callable[..., Awaitable[Principal] | Principal]


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _to_principal(user: User) -> Principal:
    return Principal(
        id=user.id,
        subject=user.subject,
        issuer=user.issuer,
        email=user.email,
        display_name=user.display_name,
    )


async def upsert_principal(
    session: AsyncSession,
    *,
    subject: str,
    issuer: str,
    email: str | None = None,
    display_name: str | None = None,
) -> Principal:
    """Record a caller's identity provider details, creating the row if new.

    Takes the fields rather than an identity object, because the object that
    carries them is the application's: a SAML assertion in juena-chatbot, and
    nothing at all where :func:`local_principal` is used.
    """

    result = await session.execute(
        select(User).where(User.issuer == issuer, User.subject == subject)
    )
    user = result.scalar_one_or_none()
    now = utc_now()
    if user is None:
        user = User(
            issuer=issuer,
            subject=subject,
            email=email,
            display_name=display_name,
            last_login_at=now,
        )
        session.add(user)
    else:
        user.email = email
        user.display_name = display_name
        user.last_login_at = now
        user.updated_at = now
    await session.flush()
    return _to_principal(user)


async def create_auth_session(session: AsyncSession, principal: Principal) -> str:
    """Issue one session token and store only its hash."""

    token = secrets.token_urlsafe(32)
    now = utc_now()
    session.add(
        AuthSession(
            token_hash=hash_session_token(token),
            user_id=principal.id,
            created_at=now,
            last_seen_at=now,
            expires_at=now + timedelta(hours=settings().SESSION_TTL_HOURS),
        )
    )
    await session.flush()
    return token


async def revoke_sessions_for_subject(session: AsyncSession, subject: str) -> int:
    """Revoke every live session for a subject.

    Used by provider-initiated single logout, where the application is told
    who to sign out but never sees their cookie.
    """

    now = utc_now()
    result = await session.execute(
        select(AuthSession)
        .join(User, User.id == AuthSession.user_id)
        .where(
            User.subject == subject,
            AuthSession.revoked_at.is_(None),
            AuthSession.expires_at > now,
        )
    )
    revoked = 0
    for auth_session in result.scalars():
        auth_session.revoked_at = now
        revoked += 1
    if revoked:
        await session.flush()
    return revoked


async def revoke_auth_session(session: AsyncSession, token: str) -> None:
    result = await session.execute(
        select(AuthSession).where(AuthSession.token_hash == hash_session_token(token))
    )
    auth_session = result.scalar_one_or_none()
    if auth_session is not None and auth_session.revoked_at is None:
        auth_session.revoked_at = utc_now()
        await session.flush()


async def session_principal(
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    session: AsyncSession = Depends(get_db_session),
) -> Principal:
    """Resolve the session cookie to a principal, or refuse the request."""

    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    now = utc_now()
    result = await session.execute(
        select(AuthSession, User)
        .join(User, User.id == AuthSession.user_id)
        .where(
            AuthSession.token_hash == hash_session_token(session_token),
            AuthSession.revoked_at.is_(None),
            AuthSession.expires_at > now,
            User.is_active.is_(True),
        )
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session"
        )
    auth_session, user = row
    auth_session.last_seen_at = now
    await session.commit()
    return _to_principal(user)


def local_principal(
    *,
    user_id: UUID,
    subject: str = "local",
    issuer: str = "local",
    email: str | None = None,
    display_name: str | None = None,
) -> PrincipalDependency:
    """A dependency returning one fixed principal, for a single-user deployment.

    The ``user_id`` is supplied by the application and must be stable across
    restarts: it is the ``users`` row, the owner of every thread, and the
    memory namespace. A value generated per process would orphan yesterday's
    conversations.

    This is injected exactly where a real provider's dependency would go, so
    adopting an institute login later replaces this callable rather than
    rewriting any route.

    Refuses to construct when the API is published (see
    :func:`refuse_published_api`) — an identity provider that authenticates
    nobody must never face a network.
    """

    principal = Principal(
        id=user_id,
        subject=subject,
        issuer=issuer,
        email=email,
        display_name=display_name,
    )

    async def dependency() -> Principal:
        return principal

    setattr(dependency, _FIXED_PRINCIPAL_ATTR, principal)
    refuse_published_api(dependency)
    return dependency


def fixed_principal(dependency: PrincipalDependency) -> Principal | None:
    """The principal behind a fixed dependency, or ``None`` for a real provider."""

    return getattr(dependency, _FIXED_PRINCIPAL_ATTR, None)


def refuse_published_api(dependency: PrincipalDependency) -> None:
    """Refuse a fixed principal paired with an API anyone else can reach.

    Called at construction — by :func:`local_principal` and again by
    ``create_app`` — rather than from a dependency, because a dependency runs
    on a request: it would fail the first call rather than refuse to boot, and
    by then the port is already open.
    """

    if fixed_principal(dependency) is None:
        return

    config = settings()
    if config.API_PUBLISHED:
        raise RuntimeError(
            "local_principal authenticates nobody and API_PUBLISHED is true. "
            "Publish only the UI; leave the API on container loopback."
        )
    if config.BIND_HOST not in _LOOPBACK_HOSTS:
        raise RuntimeError(
            f"local_principal authenticates nobody and BIND_HOST is {config.BIND_HOST!r}. "
            f"Bind the API to loopback ({', '.join(sorted(_LOOPBACK_HOSTS))})."
        )


async def ensure_principal_row(principal: Principal) -> None:
    """Create the ``users`` row a fixed principal only claims to have.

    A real identity provider writes this row as a side effect of signing
    somebody in. A fixed principal has no sign-in to hang it on, so the write
    happens at startup instead — inside ``database_lifespan``, which owns the
    session factory, and before the application serves. Without it the first
    conversation fails on ``chats.user_id``'s foreign key.
    """

    factory = get_session_factory()
    async with factory() as session:
        existing = await session.execute(
            select(User).where(User.issuer == principal.issuer, User.subject == principal.subject)
        )
        user = existing.scalar_one_or_none()
        if user is None:
            session.add(
                User(
                    id=principal.id,
                    issuer=principal.issuer,
                    subject=principal.subject,
                    email=principal.email,
                    display_name=principal.display_name,
                    last_login_at=utc_now(),
                )
            )
            logger.info("Created the users row for the configured local principal")
        elif user.id != principal.id:
            # Changing the configured id while keeping the same subject would
            # otherwise surface as a unique-constraint violation on insert,
            # which says nothing about the cause.
            raise RuntimeError(
                f"The configured principal id {principal.id} does not match the stored "
                f"id {user.id} for subject {principal.subject!r}. Restore the original "
                "id, or delete the existing user and its conversations."
            )
        await session.commit()
