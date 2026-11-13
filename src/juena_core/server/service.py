"""Build the FastAPI application: database lifespans, core's routes and ``/health``.

An application passes in what it owns: how users sign in, its own routes, its
own startup work.

Its ``service.py`` must also import two modules only for what importing them
does, or it fails later and quietly:

- the agent module, which registers the agent. Import it in ``service.py``, not
  only in ``main.py``: production starts the API with uvicorn, which never runs
  ``main.py``, and without the agent every run answers 404.
- the database models, or their tables are never created.

So an application's ``service.py`` is roughly::

    import myapp.agents.supervisor   # noqa: F401  registers the agent factory
    import myapp.server.models       # noqa: F401  adds tables to Base.metadata

    app = create_app(principal=session_principal, resume_input=ResumeInput)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from juena_core.filesystem.service import ConversationFilesystem
from juena_core.filesystem.api import build_filesystem_router
from juena_core.log import get_logger
from juena_core.schema.interrupts import ClarificationResumeInput
from juena_core.schema.server import HealthStatus
from juena_core.schema.upload_limits import UploadPolicy
from juena_core.server.agent.registry import shutdown_agents
from juena_core.server.api.endpoints import (
    DEFAULT_CLOSING_NOTE,
    ThreadActivity,
    ThreadWorkspace,
    build_api_router,
)
from juena_core.server.chat.endpoints import build_chat_router
from juena_core.server.database.checkpointer import checkpointer_lifespan
from juena_core.server.database.connection import database_lifespan
from juena_core.server.database.store import store_lifespan
from juena_core.server.identity import (
    Principal,
    PrincipalDependency,
    ensure_principal_row,
    fixed_principal,
    refuse_published_api,
)
from juena_core.server.interrupts import register_interrupt_kind
from juena_core.server.streaming.processor import StreamPolicy

logger = get_logger(__name__)

__all__ = [
    "register_interrupt_kind",
    "ThreadActivity",
    "ThreadWorkspace",
    "StreamPolicy",
    "BrowserOriginGuard",
    "create_app",
]

#: Methods that only read. A foreign page may trigger them: nothing changes,
#: and CORS keeps that page from seeing the answer.
_READ_ONLY_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class BrowserOriginGuard:
    """Refuse a browser request that changes something unless our own page sent it.

    Browsers count every host under one parent domain as the same site, so the
    session cookie's ``SameSite`` rule does not stop a page on a neighbouring
    intranet host. The ``Origin`` header does: the browser sets it, and a page
    cannot forge it.

    A plain ASGI wrapper, not ``@app.middleware("http")``, because that form
    wraps every response and the streaming routes send their own.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_origins: Collection[str],
        exempt_paths: Collection[str] = (),
    ) -> None:
        """Initialize the guard around an application.

        Args:
            app: The application to protect.
            allowed_origins: Origins whose pages may send a request that
                changes something.
            exempt_paths: Exact paths that accept such a request from any
                origin.
        """
        self.app = app
        self.allowed_origins = frozenset(allowed_origins)
        self.exempt_paths = frozenset(exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope["method"] not in _READ_ONLY_METHODS
            and scope["path"] not in self.exempt_paths
        ):
            headers = Headers(scope=scope)
            origin = headers.get("origin")
            if origin is not None:
                foreign = origin not in self.allowed_origins
            else:
                # A Python client sends neither header and is let through: it
                # has no cookie jar a website could borrow. Only a browser
                # sends Sec-Fetch-Site, and a browser always knows its origin.
                foreign = "sec-fetch-site" in headers
            if foreign:
                refusal = JSONResponse(
                    {"detail": {"code": "origin_not_allowed", "message": "Browser origin is not allowed"}},
                    status_code=403,
                )
                await refusal(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(
    *,
    principal: PrincipalDependency,
    invocation_guard: Callable[[Principal, str | None], Awaitable[None]] | None = None,
    invocation_error: Callable[[Exception], str | None] | None = None,
    resume_input: type[BaseModel] = ClarificationResumeInput,
    workspace: ThreadWorkspace | None = None,
    filesystem: ConversationFilesystem | None = None,
    browser_origins: Sequence[str] = (),
    origin_exempt_paths: Collection[str] = (),
    thread_activity: ThreadActivity | None = None,
    stream_policy: StreamPolicy | None = None,
    closing_note: str = DEFAULT_CLOSING_NOTE,
    allowed_suffixes: Collection[str] | None = None,
    upload_policy: UploadPolicy | None = None,
    run_controls: bool = False,
    title: str = "juena-core",
    version: str = "0.1.0",
    extra_routers: Sequence[APIRouter] = (),
    extra_lifespans: Sequence[Any] = (),
) -> FastAPI:
    """Build the application: lifespans, routers, and ``/health``.

    Args:
        principal: The identity dependency every route authenticates with.
            A real provider's, or :func:`~juena_core.server.identity.local_principal`.
        invocation_guard: Optional async check called with the principal and
            requested provider before a new or resumed invocation is reserved.
        invocation_error: Optional mapper for safe application-specific error messages;
            return None to retain the standard error handling.
        resume_input: The application's discriminated resume union.
        workspace: Where a thread's staged files are materialised, if anywhere.
        filesystem: Shared persistent conversation files, independent of execution.
        browser_origins: Explicit public UI/API origins allowed credentialed file
            requests and browser saves. Use the shared public origin in production.
            A browser request that changes something is refused from any other
            origin (:class:`BrowserOriginGuard`).
        origin_exempt_paths: Exact paths another website must be able to post
            to, such as the route an identity provider returns a sign-in to.
            Such a route must not rely on the session cookie for its authority.
        thread_activity: Guard shared with application routes that write thread data.
        stream_policy: Custom stream event types and silent tools.
        closing_note: Last paragraph of a staged-input manifest.
        allowed_suffixes: Legacy text upload extensions this application accepts.
        upload_policy: Optional text/binary upload policy; mutually exclusive with allowed_suffixes.
        run_controls: Add routes to read, steer and stop a running reply, and
            allow only one run per conversation at a time.
        title: Application name, shown in ``/health``.
        version: Application version, shown in ``/health``.
        extra_routers: Routers the application adds — authentication, research,
            anything core does not own. Included after core's.
        extra_lifespans: Zero-argument callables returning an async context
            manager — an ``@asynccontextmanager`` function, passed uncalled.
            Each is entered inside the database lifespan and exited before it,
            in the order given, so an application's startup work already has a
            pool and a checkpointer.

    Returns:
        The application, ready to be served by uvicorn.

    Raises:
        RuntimeError: A fixed principal is paired with a reachable API. See
            :func:`~juena_core.server.identity.refuse_published_api`.
    """

    # `local_principal` checks this when it is built too. Check again here,
    # because an application may build its principal before deciding how to serve it.
    refuse_published_api(principal)
    configured_principal = fixed_principal(principal)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Open Postgres, the checkpointer and the store, then the app's own."""

        async with database_lifespan(), checkpointer_lifespan(), store_lifespan():
            if configured_principal is not None:
                # A fixed principal never signs in, so nothing else creates its
                # `users` row, and every chat must point to one. Create it
                # before the first request arrives.
                await ensure_principal_row(configured_principal)
            try:
                async with AsyncExitStack() as stack:
                    if filesystem is not None:
                        await stack.enter_async_context(filesystem.lifespan())
                    for extra in extra_lifespans:
                        await stack.enter_async_context(extra())
                    yield
            finally:
                # Close the agents' network clients after the application's own
                # shutdown work, which may still be using an agent (background
                # jobs finishing), but while the database pools are still open.
                await shutdown_agents()

    app = FastAPI(
        title=title,
        version=version,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    if browser_origins:
        from urllib.parse import urlsplit
        from fastapi.middleware.cors import CORSMiddleware

        for origin in browser_origins:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or "*" in origin or parsed.path or parsed.query or parsed.fragment or parsed.username:
                raise ValueError("browser_origins must contain explicit HTTP origins")
        # Added before CORS so CORS stays outermost and still answers the
        # browser's pre-check (an OPTIONS request) itself.
        app.add_middleware(BrowserOriginGuard, allowed_origins=browser_origins,
                           exempt_paths=origin_exempt_paths)
        app.add_middleware(CORSMiddleware, allow_origins=list(browser_origins),
                           allow_credentials=True, allow_methods=["GET", "PUT"],
                           allow_headers=["Content-Type"])

    app.include_router(build_chat_router(principal, filesystem=filesystem))
    app.include_router(
        build_api_router(
            principal,
            invocation_guard=invocation_guard,
            invocation_error=invocation_error,
            resume_input=resume_input,
            workspace=workspace,
            filesystem=filesystem,
            thread_activity=thread_activity,
            stream_policy=stream_policy,
            closing_note=closing_note,
            allowed_suffixes=allowed_suffixes,
            upload_policy=upload_policy,
            run_controls=run_controls,
        )
    )
    if filesystem is not None:
        app.include_router(build_filesystem_router(principal, filesystem, browser_origins=tuple(browser_origins)))
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
