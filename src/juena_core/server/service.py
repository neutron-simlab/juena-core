"""The FastAPI application factory.

``create_app`` assembles what ``juena/server/service.py`` used to hard-code: the
lifespans, the two core routers, and ``/health``. Everything an application owns
— its identity provider, its extra routers, its extra lifespans, its interrupt
kinds — is passed in.

**Two side-effect-import traps live in each application's own service module,
not here**, and both fail late and quietly if forgotten:

1. *The agent registry.* Transplanted from ``juena/server/service.py:30-35``,
   which already explains it better than a new comment would:

       Imported for its side effect: the module self-registers the agent
       factory. Importing *this* module has to be enough, because the process
       serving the API is not always `main.py` -- production runs
       `uvicorn juena.server.service:app`, and the deployment plan splits the
       API and the UI into separate containers. Registering only from
       `main.py` leaves those processes with an empty registry, so every
       invocation would 404.

2. *The database models.* A model class never imported is never created by
   ``create_all``, and it fails at first write rather than at startup.

Both mitigations are the same: import the module in the application's
``service.py`` with ``# noqa: F401`` and a comment. Core cannot do it — it does
not know the module names — but it does make the first failure loud:
:func:`~juena_core.server.agent.registry.get_default_agent` raises when nothing
registered a default, and ``test_core_owns_exactly_three_tables`` pins the
second.

So an application's ``service.py`` is roughly::

    import myapp.agents.supervisor   # noqa: F401  registers the agent factory
    import myapp.server.models       # noqa: F401  adds tables to Base.metadata

    app = create_app(principal=session_principal, resume_input=ResumeInput)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Collection, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel

from juena_core.log import get_logger
from juena_core.schema.interrupts import ClarificationResumeInput
from juena_core.schema.server import HealthStatus
from juena_core.server.agent.registry import shutdown_agents
from juena_core.server.api.endpoints import DEFAULT_CLOSING_NOTE, ThreadWorkspace, build_api_router
from juena_core.server.chat.endpoints import build_chat_router
from juena_core.server.database.checkpointer import checkpointer_lifespan
from juena_core.server.database.connection import database_lifespan
from juena_core.server.database.store import store_lifespan
from juena_core.server.identity import (
    PrincipalDependency,
    ensure_principal_row,
    fixed_principal,
    refuse_published_api,
)
from juena_core.server.interrupts import register_interrupt_kind
from juena_core.server.streaming.processor import StreamPolicy

logger = get_logger(__name__)

__all__ = ["register_interrupt_kind", "ThreadWorkspace", "StreamPolicy", "create_app"]


def create_app(
    *,
    principal: PrincipalDependency,
    resume_input: type[BaseModel] = ClarificationResumeInput,
    workspace: ThreadWorkspace | None = None,
    stream_policy: StreamPolicy | None = None,
    closing_note: str = DEFAULT_CLOSING_NOTE,
    allowed_suffixes: Collection[str] | None = None,
    title: str = "juena-core",
    version: str = "0.1.0",
    extra_routers: Sequence[APIRouter] = (),
    extra_lifespans: Sequence[Any] = (),
) -> FastAPI:
    """Build the application: lifespans, routers, and ``/health``.

    Args:
        principal: The identity dependency every route authenticates with.
            A real provider's, or :func:`~juena_core.server.identity.local_principal`.
        resume_input: The application's discriminated resume union.
        workspace: Where a thread's staged files are materialised, if anywhere.
        stream_policy: Custom stream event types and silent tools.
        closing_note: Last paragraph of a staged-input manifest.
        allowed_suffixes: Upload extensions this application accepts.
        title: Application name, shown in ``/health``.
        version: Application version, shown in ``/health``.
        extra_routers: Routers the application adds — authentication, research,
            anything core does not own. Included after core's.
        extra_lifespans: Zero-argument callables returning an async context
            manager — an ``@asynccontextmanager`` function, passed uncalled.
            Each is entered inside the database lifespan and exited before it,
            in the order given, so an application's startup work already has a
            pool and a checkpointer.

    Raises:
        RuntimeError: A fixed principal is paired with a reachable API. See
            :func:`~juena_core.server.identity.refuse_published_api`.
    """

    # The second of two gates. `local_principal` already refuses at
    # construction, but an application may build its principal before it
    # decides how to serve it, and this is the moment the port is about to
    # open. Both checks are cheap; only this one sees the finished application.
    refuse_published_api(principal)
    configured_principal = fixed_principal(principal)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Open Postgres, the checkpointer and the store, then the app's own."""

        async with database_lifespan(), checkpointer_lifespan(), store_lifespan():
            if configured_principal is not None:
                # A fixed principal has no sign-in to create its `users` row as
                # a side effect, and `chats.user_id` is a foreign key. Inside
                # the database lifespan, so the session factory exists; before
                # serving, so the first conversation cannot lose the race.
                await ensure_principal_row(configured_principal)
            async with AsyncExitStack() as stack:
                for extra in extra_lifespans:
                    await stack.enter_async_context(extra())
                try:
                    yield
                finally:
                    # Agents may hold network clients for the process lifetime;
                    # close them before the pools they may be using go away.
                    await shutdown_agents()

    app = FastAPI(
        title=title,
        version=version,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.include_router(build_chat_router(principal))
    app.include_router(
        build_api_router(
            principal,
            resume_input=resume_input,
            workspace=workspace,
            stream_policy=stream_policy,
            closing_note=closing_note,
            allowed_suffixes=allowed_suffixes,
        )
    )
    for router in extra_routers:
        app.include_router(router)

    @app.get("/health")
    async def health_check() -> HealthStatus:
        """Health check endpoint."""

        return HealthStatus(
            status="ok",
            version=version,
            details={"service": title, "uptime": "running"},
        )

    return app
