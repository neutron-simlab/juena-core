"""Runtime identity resolution and tenant-scoped specialist backends."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.utils import file_data_to_string
from langgraph.config import get_config
from langgraph.runtime import get_runtime

from juena_core.agents.backends import (
    FINDINGS_PREFIX,
    FindingsStateBackend,
    ReadOnlyFilesystemBackend,
)
from juena_core.config import settings as core_settings
from juena_core.runtime_context import _context_value
from juena_core.sandbox.backend import PodmanSandboxBackend, ReadOnlySandboxInputsBackend
from juena_core.sandbox.config import sandbox_enabled, sandbox_settings
from juena_core.sandbox.jobs import SandboxJobs, tenant_id_for, workspace_id_for
from juena_core.sandbox.workspace import WorkspaceStore

__all__ = [
    "RuntimePodmanSandboxBackend",
    "sandbox_enabled",
    "get_workspace_store",
    "set_workspace_store_for_tests",
    "get_sandbox_jobs",
    "set_sandbox_jobs_for_tests",
    "close_sandbox_jobs",
    "sandbox_lifespan",
    "build_sandbox_backend",
    "stage_runtime_inputs",
    "delete_runtime_workspace",
]


@dataclass(frozen=True, slots=True)
class _SandboxIdentity:
    user_id: str
    thread_id: str
    run_id: str | None
    tenant_id: str
    workspace_id: str

    @classmethod
    def from_runtime(cls, runtime: Any) -> "_SandboxIdentity":
        context = getattr(runtime, "context", None)
        user_id = _context_value(context, "user_id")
        thread_id = _context_value(context, "thread_id")
        if not user_id or not thread_id:
            raise RuntimeError("Sandbox execution requires authenticated user_id and thread_id")
        config = getattr(runtime, "config", None)
        if config is None:
            try:
                config = get_config()
            except (KeyError, RuntimeError):
                config = {}
        execution_info = getattr(runtime, "execution_info", None)
        run_id = getattr(execution_info, "run_id", None)
        if not run_id:
            run_id = config.get("run_id") if hasattr(config, "get") else None
        secret = sandbox_settings().identity_secret
        if not secret:
            raise RuntimeError("Sandbox identity secret is unavailable")
        return cls(
            user_id=user_id,
            thread_id=thread_id,
            run_id=str(run_id) if run_id else None,
            tenant_id=tenant_id_for(user_id, secret),
            workspace_id=workspace_id_for(user_id, thread_id, secret),
        )


_workspace_store: WorkspaceStore | None = None


def get_workspace_store() -> WorkspaceStore:
    global _workspace_store
    config = sandbox_settings()
    if _workspace_store is None or _workspace_store.root != config.workspace_root:
        _workspace_store = WorkspaceStore(
            config.workspace_root,
            limit_bytes=config.workspace_limit_bytes,
            idle_ttl_seconds=config.idle_ttl_seconds,
        )
    return _workspace_store


def set_workspace_store_for_tests(store: WorkspaceStore | None) -> None:
    global _workspace_store
    _workspace_store = store


# The application's shared connection pool. It lives here rather than in
# `jobs.py` so that the standalone worker, which imports `jobs.py`, never pulls
# in — or has to satisfy — the application's configuration.
_jobs: SandboxJobs | None = None
_jobs_lock = threading.Lock()


def get_sandbox_jobs() -> SandboxJobs:
    global _jobs
    if _jobs is not None:
        return _jobs
    config = sandbox_settings()
    if not config.enabled:
        raise RuntimeError("Sandbox execution is disabled")
    database_url = core_settings().DATABASE_URL
    if not database_url:
        raise RuntimeError("Sandbox jobs need DATABASE_URL")
    with _jobs_lock:
        if _jobs is None:
            jobs = SandboxJobs(
                database_url,
                concurrency=config.concurrency,
            )
            jobs.open()
            _jobs = jobs
    return _jobs


def set_sandbox_jobs_for_tests(jobs: SandboxJobs | None) -> None:
    global _jobs
    _jobs = jobs


def close_sandbox_jobs() -> None:
    global _jobs
    if _jobs is not None:
        _jobs.close()
        _jobs = None


@asynccontextmanager
async def sandbox_lifespan() -> AsyncGenerator[None, None]:
    """Close the lazily opened application-side queue during API shutdown."""

    try:
        yield
    finally:
        close_sandbox_jobs()


class RuntimePodmanSandboxBackend(BaseSandbox):
    """Resolve the tenant-bound Podman sandbox from the active graph runtime."""

    @staticmethod
    def _resolve() -> PodmanSandboxBackend:
        runtime = get_runtime()
        identity = _SandboxIdentity.from_runtime(runtime)
        return PodmanSandboxBackend(
            jobs=get_sandbox_jobs(),
            workspaces=get_workspace_store(),
            tenant_id=identity.tenant_id,
            workspace_id=identity.workspace_id,
            user_id=identity.user_id,
            thread_id=identity.thread_id,
            run_id=identity.run_id,
        )

    @property
    def id(self) -> str:
        return self._resolve().id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self._resolve().execute(command, timeout=timeout)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._resolve().upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._resolve().download_files(paths)


def build_sandbox_backend(
    *,
    repo_cache_root: Path | None,
    skills_dir: Path | None,
    has_skills: bool,
) -> CompositeBackend:
    if not sandbox_enabled():
        raise RuntimeError("Cannot build a sandbox backend while SANDBOX_ENABLED is false")

    sandbox = RuntimePodmanSandboxBackend()
    routes: dict[str, Any] = {
        "/inputs/": ReadOnlySandboxInputsBackend(sandbox),
        # Graph state, never the container. The sandbox root is read-only, so a
        # specialist writing here got `Errno 30: Read-only file system`; and even
        # a writable container would hide findings from a background specialist,
        # which runs with no sandbox at all.
        FINDINGS_PREFIX: FindingsStateBackend(),
    }
    if repo_cache_root is not None:
        routes["/repos/"] = ReadOnlyFilesystemBackend(
            root_dir=repo_cache_root,
            virtual_mode=True,
            label="read-only repository",
        )
    if skills_dir is not None and has_skills:
        routes["/skills/"] = ReadOnlyFilesystemBackend(
            root_dir=skills_dir,
            virtual_mode=True,
            label="read-only skill",
        )
    return CompositeBackend(
        default=sandbox,
        routes=routes,
        # Deep Agents stores oversized tool output and conversation summaries
        # beneath this root. The sandbox root filesystem is intentionally
        # read-only; `/workspace` is its only writable area.
        artifacts_root="/workspace",
    )


async def stage_runtime_inputs(
    *,
    user_id: str,
    thread_id: str,
    files: dict[str, Any],
) -> None:
    if not sandbox_enabled():
        return
    secret = sandbox_settings().identity_secret
    if not secret:
        raise RuntimeError("Sandbox identity secret is unavailable")
    tenant_id = tenant_id_for(user_id, secret)
    workspace_id = workspace_id_for(user_id, thread_id, secret)
    staged: list[tuple[str, bytes]] = []
    for path, file_data in sorted(files.items()):
        if not path.startswith("/inputs/") or file_data is None:
            continue
        staged.append((path, file_data_to_string(file_data).encode("utf-8")))
    jobs = get_sandbox_jobs()

    def stage() -> None:
        with jobs.workspace_guard(workspace_id) as available:
            if not available:
                raise RuntimeError("Sandbox workspace is busy")
            get_workspace_store().stage_inputs(workspace_id, tenant_id, staged)

    await asyncio.to_thread(stage)


async def delete_runtime_workspace(*, user_id: str, thread_id: str) -> None:
    if not sandbox_enabled():
        return
    secret = sandbox_settings().identity_secret
    if not secret:
        raise RuntimeError("Sandbox identity secret is unavailable")
    tenant_id = tenant_id_for(user_id, secret)
    workspace_id = workspace_id_for(user_id, thread_id, secret)
    jobs = get_sandbox_jobs()

    def delete() -> None:
        with jobs.workspace_guard(workspace_id) as available:
            if available:
                get_workspace_store().delete(workspace_id, tenant_id)

    await asyncio.to_thread(delete)
