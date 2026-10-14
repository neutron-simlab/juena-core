"""Persistent conversation workspace and expiring scratch storage."""

from __future__ import annotations

import asyncio
import os
import fcntl
import hashlib
import hmac
import re
import shutil
from contextlib import asynccontextmanager, contextmanager, ExitStack, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from time import monotonic, sleep, time
from uuid import uuid4

from juena_core.artifacts import DOWNLOAD_TYPES, MAX_ARTIFACT_BYTES, MAX_FILE_ARTIFACT_BYTES
from juena_core.log import get_logger

logger = get_logger(__name__)

COLLECTED_EXTENSIONS = frozenset(DOWNLOAD_TYPES) | {".png"}
MAX_FILE_ARTIFACTS_PER_EXECUTION = 8
MAX_ARTIFACTS_PER_EXECUTION = 4

__all__ = [
    "WorkspaceBusy",
    "RevisionConflict",
    "tenant_id_for",
    "workspace_id_for",
    "MAX_TRANSFER_BYTES",
    "OWNER_FILE",
    "WorkspaceFileResult",
    "WorkspaceArtifactScan",
    "WorkspaceOwnershipError",
    "WorkspaceStore",
    "assert_safe_path",
]

MAX_TRANSFER_BYTES = 100 * 1024 * 1024
OWNER_FILE = ".owner"
DIRECTORY_MODE = 0o2770
FILE_MODE = 0o660


@dataclass(frozen=True, slots=True)
class WorkspaceFileResult:
    path: str
    content: bytes | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceArtifactScan:
    """Accepted outputs plus a bounded account of files the worker refused."""

    accepted: tuple[str, ...] = ()
    dropped: tuple[tuple[str, str], ...] = ()
    omitted_drops: int = 0


class WorkspaceOwnershipError(RuntimeError):
    pass


def assert_safe_path(path: str, message: str = "invalid_path") -> PurePosixPath:
    """Reject anything that is not a plain absolute path inside the sandbox."""

    posix = PurePosixPath(path)
    if not path.startswith("/") or path.startswith("//") or "\x00" in path or "\\" in path or ".." in posix.parts or "~" in posix.parts:
        raise ValueError(message)
    return posix


class WorkspaceStore:
    def __init__(
        self,
        root: Path,
        *,
        limit_bytes: int = 5 * 1024**3,
        idle_ttl_seconds: int = 24 * 3600,
    ) -> None:
        self.root = root.resolve()
        self.limit_bytes = limit_bytes
        self.idle_ttl_seconds = idle_ttl_seconds

    def workspace_root(self, workspace_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", workspace_id):
            raise ValueError("invalid workspace identity")
        return self.root / workspace_id

    def workspace_dir(self, workspace_id: str) -> Path:
        return self.workspace_root(workspace_id) / "workspace"

    def inputs_dir(self, workspace_id: str) -> Path:
        return self.workspace_root(workspace_id) / "inputs"

    @staticmethod
    def _make_dir(path: Path) -> Path:
        """Create a setgid directory independent of the caller's umask."""
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise ValueError("Symlink directories are not allowed")
        path.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            path.chmod(DIRECTORY_MODE)
        return path

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        path.write_bytes(content)
        with suppress(OSError):
            path.chmod(FILE_MODE)

    def ensure(self, workspace_id: str, tenant_id: str) -> None:
        root = self.workspace_root(workspace_id)
        if self.deleted(workspace_id):
            raise WorkspaceDeleted("Conversation workspace was deleted")
        if root.is_symlink():
            raise WorkspaceOwnershipError("Sandbox workspace root cannot be a symlink")
        self._make_dir(root)
        owner = root / OWNER_FILE
        if owner.is_symlink():
            raise WorkspaceOwnershipError("Sandbox workspace owner marker is invalid")
        if owner.exists():
            if owner.read_text(encoding="ascii").strip() != tenant_id:
                raise WorkspaceOwnershipError("Sandbox workspace belongs to another tenant")
        else:
            self._write(owner, f"{tenant_id}\n".encode("ascii"))
        self._make_dir(self.workspace_dir(workspace_id))
        self._make_dir(self.inputs_dir(workspace_id))
        self._make_dir(self.workspace_dir(workspace_id) / ".scratch")
        os.utime(root, None)

    def ensure_outputs(self, workspace_id: str) -> Path:
        outputs = self.workspace_dir(workspace_id) / "outputs"
        if outputs.is_symlink() or (outputs.exists() and not outputs.is_dir()):
            outputs.unlink(missing_ok=True)
        return self._make_dir(outputs)

    @staticmethod
    def resolve_path(workspace: Path, inputs: Path, path: str, *, write: bool) -> Path:
        posix = assert_safe_path(path)
        if path == "/workspace" or path.startswith("/workspace/"):
            relative = posix.relative_to("/workspace")
            root = workspace
        elif not write and (path == "/inputs" or path.startswith("/inputs/")):
            relative = posix.relative_to("/inputs")
            root = inputs
        else:
            raise PermissionError("permission_denied")
        target = root / Path(*relative.parts)
        for component in (target, *target.parents):
            if component.is_symlink():
                raise ValueError("invalid_path: symlinks are not allowed")
        if target.exists() and not (target.is_dir() or target.is_file()):
            raise ValueError("Only regular files and directories are allowed")
        if target.is_file() and target.stat().st_nlink != 1:
            raise ValueError("Hard-linked files are not allowed")
        return target

    def deleted(self, workspace_id: str) -> bool:
        self.workspace_root(workspace_id)
        return (self.root / ".locks" / f"{workspace_id}.deleted").exists()

    def check_owner(self, workspace_id: str, tenant_id: str) -> None:
        root = self.workspace_root(workspace_id)
        if self.deleted(workspace_id):
            raise WorkspaceDeleted("Conversation workspace was deleted")
        owner = root / OWNER_FILE
        if root.is_symlink() or owner.is_symlink() or owner.read_text(encoding="ascii").strip() != tenant_id:
            raise WorkspaceOwnershipError("Workspace belongs to another tenant")

    @contextmanager
    def _lock(self, workspace_id: str, name: str, *, shared: bool, timeout: float | None):
        self.workspace_root(workspace_id)  # validate before forming a lock filename
        locks = self._make_dir(self.root / ".locks")
        path = locks / f"{workspace_id}.{name}"
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, FILE_MODE)
        try:
            with suppress(OSError):
                os.fchmod(fd, FILE_MODE)
            deadline = monotonic() + timeout if timeout is not None else None
            while True:
                try:
                    fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if deadline is not None and monotonic() >= deadline:
                        raise WorkspaceBusy("Conversation files are busy; try again when the operation finishes") from None
                    sleep(0.01)
            yield
        finally:
            os.close(fd)

    @contextmanager
    def guard(self, workspace_id: str, *, operation: bool = False, shared: bool = False,
              blocking: bool = False, execution: bool = False, timeout: float = 5):
        """Readers share files; short writes wait. Commands have a separate busy gate."""
        with ExitStack() as stack:
            if not operation:
                try:
                    stack.enter_context(self._lock(workspace_id, "execution", shared=not execution,
                                                   timeout=None if blocking else 0))
                except WorkspaceBusy as exc:
                    raise WorkspaceExecuting("Files are busy while a sandbox command is running") from exc
            stack.enter_context(self._lock(workspace_id, "operation" if operation else "files", shared=shared,
                                           timeout=None if blocking else (0 if operation and not shared else timeout)))
            yield

    async def acquire(self, workspace_id: str, **kwargs):
        """Acquire without blocking the event loop or leaving a cancelled waiter behind."""
        deadline = monotonic() + (0.25 if kwargs.get("operation") and not kwargs.get("shared") else 5)
        while True:
            lease = self.guard(workspace_id, timeout=0, **kwargs)
            try:
                lease.__enter__()
                return lease
            except WorkspaceExecuting:
                raise
            except WorkspaceBusy:
                if monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.01)

    @asynccontextmanager
    async def aguard(self, workspace_id: str, **kwargs):
        lease = await self.acquire(workspace_id, **kwargs)
        try:
            yield
        finally:
            lease.__exit__(None, None, None)

    def paused(self, workspace_id: str) -> bool:
        return (self.workspace_root(workspace_id) / ".paused").exists()

    def mark_paused(self, workspace_id: str, paused: bool) -> None:
        marker = self.workspace_root(workspace_id) / ".paused"
        if paused:
            self._write(marker, b"pending interrupt\n")
        else:
            marker.unlink(missing_ok=True)

    def put(self, workspace_id: str, path: str, content: bytes, *, revision: str | None = None, create: bool = False) -> str:
        """Atomic write; the caller holds the files guard."""
        target = self.resolve_path(self.workspace_dir(workspace_id), self.inputs_dir(workspace_id), path, write=True)
        if create and target.exists():
            raise FileExistsError(path)
        if revision is not None:
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != revision:
                raise RevisionConflict("This file changed. Reload it before saving; your draft has been kept.")
        old_size = target.stat().st_size if target.is_file() else 0
        usage = self.workspace_root(workspace_id) / ".usage"
        size = int(usage.read_text()) if usage.exists() else self._directory_size(self.workspace_dir(workspace_id))
        new_size = size - old_size + len(content)
        if new_size > self.limit_bytes:
            raise ValueError("Conversation workspace quota exceeded")
        # A crash between mutation and accounting forces a fresh scan next time.
        self.invalidate_usage(workspace_id)
        self._make_dir(target.parent)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, FILE_MODE)
            with os.fdopen(fd, "wb") as stream:
                with suppress(OSError):
                    os.fchmod(stream.fileno(), FILE_MODE)
                stream.write(content)
            os.replace(temporary, target)
            self._record_usage(workspace_id, new_size)
        finally:
            temporary.unlink(missing_ok=True)
        os.utime(self.workspace_root(workspace_id), None)
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _directory_size(root: Path) -> int:
        return sum(
            path.stat().st_size
            for path in root.rglob("*")
            if not path.is_symlink() and path.is_file()
        )

    def enforce_limit(self, workspace_id: str) -> None:
        size = self._directory_size(self.workspace_dir(workspace_id))
        self._record_usage(workspace_id, size)
        if size > self.limit_bytes:
            raise RuntimeError(
                f"Sandbox workspace exceeds its {self.limit_bytes}-byte limit"
            )

    def invalidate_usage(self, workspace_id: str) -> None:
        (self.workspace_root(workspace_id) / ".usage").unlink(missing_ok=True)

    def _record_usage(self, workspace_id: str, size: int) -> None:
        target = self.workspace_root(workspace_id) / ".usage"
        temporary = target.with_suffix(".tmp")
        self._write(temporary, str(size).encode("ascii"))
        temporary.replace(target)

    def stage_inputs(
        self,
        workspace_id: str,
        tenant_id: str,
        files: list[tuple[str, bytes]],
    ) -> None:
        if sum(len(content) for _path, content in files) > MAX_TRANSFER_BYTES:
            raise ValueError("Staged inputs exceed the 100 MB limit")
        self.ensure(workspace_id, tenant_id)
        inputs = self.inputs_dir(workspace_id)
        replacement = self._make_dir(inputs.with_name(f"inputs-{uuid4().hex}.tmp"))
        try:
            for path, content in files:
                if not path.startswith("/inputs/"):
                    raise ValueError("Only /inputs files may be staged")
                target = self.resolve_path(
                    self.workspace_dir(workspace_id), replacement, path, write=False
                )
                self._make_dir(target.parent)
                self._write(target, content)
            old = inputs.with_name(f"inputs-{uuid4().hex}.old")
            os.replace(inputs, old)
            os.replace(replacement, inputs)
            shutil.rmtree(old, ignore_errors=True)
        finally:
            shutil.rmtree(replacement, ignore_errors=True)

    def upload(
        self,
        workspace_id: str,
        tenant_id: str,
        files: list[tuple[str, bytes]],
    ) -> list[WorkspaceFileResult]:
        self.ensure(workspace_id, tenant_id)
        results: list[WorkspaceFileResult] = []
        transferred = 0
        for path, content in files:
            try:
                transferred += len(content)
                if transferred > MAX_TRANSFER_BYTES:
                    raise ValueError("Uploaded files exceed the 100 MB transfer limit")
                target = self.resolve_path(
                    self.workspace_dir(workspace_id),
                    self.inputs_dir(workspace_id),
                    path,
                    write=True,
                )
                if target.exists():
                    results.append(WorkspaceFileResult(path, error="file already exists"))
                    continue
                self.put(workspace_id, path, content, create=True)
                results.append(WorkspaceFileResult(path))
            except (OSError, PermissionError, ValueError) as exc:
                results.append(WorkspaceFileResult(path, error=str(exc)))
        self.enforce_limit(workspace_id)
        return results

    def download(
        self,
        workspace_id: str,
        tenant_id: str,
        paths: list[str],
    ) -> list[WorkspaceFileResult]:
        self.ensure(workspace_id, tenant_id)
        results: list[WorkspaceFileResult] = []
        transferred = 0
        for path in paths:
            try:
                target = self.resolve_path(
                    self.workspace_dir(workspace_id),
                    self.inputs_dir(workspace_id),
                    path,
                    write=False,
                )
                if not target.is_file() or target.is_symlink():
                    results.append(WorkspaceFileResult(path, error="file_not_found"))
                    continue
                size = target.stat().st_size
                if transferred + size > MAX_TRANSFER_BYTES:
                    results.append(
                        WorkspaceFileResult(path, error="download transfer limit exceeded")
                    )
                    continue
                transferred += size
                results.append(WorkspaceFileResult(path, content=target.read_bytes()))
            except (OSError, PermissionError, ValueError) as exc:
                results.append(WorkspaceFileResult(path, error=str(exc)))
        return results

    def artifacts_since(
        self,
        workspace_id: str,
        started_at: datetime,
    ) -> WorkspaceArtifactScan:
        """Return accepted outputs and bounded rejection details for one execution."""
        output_dir = self.workspace_dir(workspace_id) / "outputs"
        if output_dir.is_symlink() or not output_dir.is_dir():
            return WorkspaceArtifactScan()
        output_root = output_dir.resolve()
        threshold = int(started_at.timestamp() * 1_000_000_000)
        names: list[str] = []
        dropped: list[tuple[str, str]] = []
        omitted_drops = 0
        images = 0

        def reject(name: str, reason: str) -> None:
            nonlocal omitted_drops
            if len(dropped) < MAX_FILE_ARTIFACTS_PER_EXECUTION:
                dropped.append((name, reason))
            else:
                omitted_drops += 1

        for path in sorted(output_dir.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            if any(parent.is_symlink() for parent in path.parents) or path.stat().st_nlink != 1:
                continue
            name = path.relative_to(output_dir).as_posix()
            stat = path.stat()
            if stat.st_mtime_ns < threshold:
                continue
            extension = path.suffix.lower()
            if extension not in COLLECTED_EXTENSIONS:
                reject(name, "Generated file type is not allowed")
                continue
            if extension == ".png":
                if stat.st_size > MAX_ARTIFACT_BYTES:
                    reject(name, "Generated PNG exceeds the 5 MB limit")
                    continue
                if images >= MAX_ARTIFACTS_PER_EXECUTION:
                    reject(
                        name,
                        f"This execution already has the maximum of "
                        f"{MAX_ARTIFACTS_PER_EXECUTION} plots",
                    )
                    continue
            elif stat.st_size > MAX_FILE_ARTIFACT_BYTES:
                reject(name, "Generated file exceeds the 10 MB limit")
                continue
            if len(names) >= MAX_FILE_ARTIFACTS_PER_EXECUTION:
                reject(
                    name,
                    f"This execution already has the maximum of "
                    f"{MAX_FILE_ARTIFACTS_PER_EXECUTION} generated files",
                )
                continue
            names.append(name)
            if extension == ".png":
                images += 1
        return WorkspaceArtifactScan(tuple(names), tuple(dropped), omitted_drops)

    def delete(self, workspace_id: str, tenant_id: str) -> None:
        root = self.workspace_root(workspace_id)
        if root.is_symlink():
            raise WorkspaceOwnershipError("Workspace root cannot be a symlink")
        if root.exists():
            owner = root / OWNER_FILE
            if owner.is_symlink() or owner.read_text(encoding="ascii").strip() != tenant_id:
                raise WorkspaceOwnershipError("Workspace belongs to another tenant")
        # Keep this outside the deleted directory: a queued worker must not recreate it.
        marker = self._make_dir(self.root / ".locks") / f"{workspace_id}.deleted"
        previous = marker.read_bytes() if marker.exists() else None
        self._write(marker, b"deleted\n")
        try:
            if root.exists():
                self._remove_tree(root)
        except OSError:
            if previous is None:
                marker.unlink(missing_ok=True)
            else:
                self._write(marker, previous)
            raise

    @staticmethod
    def _remove_tree(root: Path) -> None:
        # Containers may leave owned directories without write/search permission.
        # Walk without following links; chmod only the directory entries we own.
        with suppress(PermissionError):
            root.chmod(DIRECTORY_MODE)
        for parent, directories, _files in os.walk(root):
            for name in directories:
                path = Path(parent) / name
                if not path.is_symlink():
                    with suppress(PermissionError):
                        path.chmod(DIRECTORY_MODE)
        shutil.rmtree(root)

    def cleanup_expired(self, active_workspaces: set[str]) -> int:
        cutoff = time() - self.idle_ttl_seconds
        removed = 0
        if not self.root.is_dir():
            return 0
        for candidate in self.root.iterdir():
            try:
                if (candidate.name.startswith(".") or candidate.is_symlink() or not candidate.is_dir()
                    or candidate.name in active_workspaces or candidate.stat().st_mtime >= cutoff):
                    continue
                with self.guard(candidate.name, operation=True), self.guard(candidate.name, timeout=0):
                    if self.paused(candidate.name) or candidate.stat().st_mtime >= cutoff:
                        continue
                    scratch = self.workspace_dir(candidate.name) / ".scratch"
                    if scratch.is_dir() and not scratch.is_symlink():
                        self.invalidate_usage(candidate.name)
                        self._remove_tree(scratch)
                        removed += 1
            except WorkspaceBusy:
                continue
            except OSError:
                logger.exception("Could not clean scratch for workspace %s", candidate.name)
        return removed


class WorkspaceBusy(RuntimeError):
    pass


class WorkspaceExecuting(WorkspaceBusy):
    pass


class WorkspaceDeleted(WorkspaceOwnershipError):
    pass


class RevisionConflict(ValueError):
    pass


def tenant_id_for(user_id: str, secret: str) -> str:
    return hmac.new(secret.encode(), user_id.encode(), hashlib.sha256).hexdigest()


def workspace_id_for(user_id: str, thread_id: str, secret: str) -> str:
    return hmac.new(secret.encode(), f"{user_id}\0{thread_id}".encode(), hashlib.sha256).hexdigest()
