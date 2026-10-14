"""One path namespace for conversation files, independent of execution."""

from __future__ import annotations

__all__ = ['ConversationFilesystem', 'ThreadFiles', 'configure_filesystem', 'get_filesystem', 'MAX_EDIT_BYTES', 'PREVIEW_BYTES', 'SCRATCH']

import asyncio
import hashlib
import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from deepagents.backends.utils import file_data_to_string

from juena_core.filesystem.storage import WorkspaceBusy, WorkspaceStore, assert_safe_path, tenant_id_for, workspace_id_for
from juena_core.log import get_logger

logger = get_logger(__name__)

MAX_EDIT_BYTES = 1024 * 1024
PREVIEW_BYTES = 64 * 1024
SCRATCH = "/workspace/.scratch"


class ConversationFilesystem:
    def __init__(self, store: WorkspaceStore, *, identity_secret: str,
                 active_workspaces: Callable[[], set[str]] | None = None) -> None:
        if len(identity_secret) < 32:
            raise ValueError("Filesystem identity secret must contain at least 32 characters")
        self.store = store
        self.identity_secret = identity_secret
        self.active_workspaces = active_workspaces or set

    def assert_idle(self, workspace_id: str) -> None:
        # Durable queues can outlive the process that held the operation lock.
        if workspace_id in self.active_workspaces():
            raise WorkspaceBusy("Conversation has a queued or running execution")

    @asynccontextmanager
    async def lifespan(self):
        async def clean():
            while True:
                try:
                    active = await asyncio.to_thread(self.active_workspaces)
                    await asyncio.to_thread(self.store.cleanup_expired, active)
                except Exception:
                    logger.exception("Conversation scratch cleanup failed")
                await asyncio.sleep(3600)

        task = asyncio.create_task(clean())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def for_thread(self, user_id: str, thread_id: str) -> ThreadFiles:
        if not user_id or not thread_id:
            raise ValueError("Authenticated user and conversation are required")
        return ThreadFiles(self.store, workspace_id_for(user_id, thread_id, self.identity_secret),
                           tenant_id_for(user_id, self.identity_secret))


class ThreadFiles:
    def __init__(self, store: WorkspaceStore, workspace_id: str, tenant_id: str) -> None:
        self.store, self.workspace_id, self.tenant_id = store, workspace_id, tenant_id

    def ensure(self) -> None:
        self.store.ensure(self.workspace_id, self.tenant_id)

    def guard(self, **kwargs):
        return self.store.guard(self.workspace_id, **kwargs)

    def aguard(self, **kwargs):
        return self.store.aguard(self.workspace_id, **kwargs)

    async def acquire(self, **kwargs):
        return await self.store.acquire(self.workspace_id, **kwargs)

    def path(self, path: str, *, write: bool = False) -> Path:
        return self.store.resolve_path(self.store.workspace_dir(self.workspace_id),
                                       self.store.inputs_dir(self.workspace_id), path, write=write)

    def aliases(self, *, live_only: bool = False) -> dict[str, str]:
        path = self.store.workspace_root(self.workspace_id) / ".uploads.json"
        aliases = json.loads(path.read_text()) if path.exists() else {}
        if live_only:
            return {old: new for old, new in aliases.items() if self.path(new).is_file()}
        return aliases

    def needs_migration(self, files: dict[str, Any]) -> bool:
        if not self.store.workspace_dir(self.workspace_id).is_dir():
            return True
        aliases = self.aliases()
        return any(path.startswith("/inputs/uploads/") and data is not None and path not in aliases
                   for path, data in files.items())

    def migrate(self, files: dict[str, Any]) -> dict[str, str]:
        """Called under the files lock. A saved mapping survives subsequent user edits."""
        self.ensure()
        aliases = self.aliases()
        changed = False
        for legacy, data in sorted(files.items()):
            if not legacy.startswith("/inputs/uploads/") or data is None or legacy in aliases:
                continue
            content = file_data_to_string(data).encode("utf-8")
            name = PurePosixPath(legacy).name
            candidate = f"/workspace/uploads/{name}"
            index = 2
            while self.path(candidate).exists() and self.path(candidate).read_bytes() != content:
                candidate = f"/workspace/uploads/{Path(name).stem}_legacy{index}{Path(name).suffix}"
                index += 1
            if not self.path(candidate).exists():
                self.store.put(self.workspace_id, candidate, content, create=True)
            aliases[legacy] = candidate
            changed = True
        if changed:
            target = self.store.workspace_root(self.workspace_id) / ".uploads.json"
            temporary = target.with_suffix(".tmp")
            self.store._write(temporary, json.dumps(aliases).encode())
            temporary.replace(target)
        return aliases

    def add_upload(self, name: str, content: bytes) -> str:
        candidate = f"/workspace/uploads/{name}"
        index = 2
        while self.path(candidate).exists():
            candidate = f"/workspace/uploads/{Path(name).stem}_{index}{Path(name).suffix}"
            index += 1
        self.store.put(self.workspace_id, candidate, content, create=True)
        return candidate

    def read(self, path: str, state_files: dict[str, Any], *, limit: int | None = None) -> bytes:
        self.store.check_owner(self.workspace_id, self.tenant_id)
        assert_safe_path(path)
        path = str(PurePosixPath(path))
        alias = self.aliases().get(path)
        if alias:
            path = alias
        if path.startswith(("/inputs/", "/findings/")):
            data = state_files.get(path)
            if data is None:
                raise FileNotFoundError(path)
            content = file_data_to_string(data).encode("utf-8")
            return content if limit is None else content[:limit]
        target = self.path(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        with target.open("rb") as stream:
            return stream.read(-1 if limit is None else limit)

    def list(self, path: str, state_files: dict[str, Any], *, hidden: bool = False,
             offset: int = 0, limit: int = 100) -> dict[str, Any]:
        self.store.check_owner(self.workspace_id, self.tenant_id)
        assert_safe_path(path)
        path = str(PurePosixPath(path))
        entries = []
        if path == "/":
            entries = [{"path": p, "is_dir": True, "size": 0, "writable": p == "/workspace"}
                       for p in ("/workspace", "/inputs", "/findings")]
        elif path == "/workspace" or path.startswith("/workspace/"):
            directory = self.path(path)
            if not directory.is_dir():
                raise FileNotFoundError(path)
            for item in directory.iterdir():
                virtual = f"{path}/{item.name}"
                if not hidden and item.name.startswith("."):
                    continue
                try:
                    self.path(virtual)
                except (ValueError, PermissionError):
                    continue
                stat = item.stat()
                entries.append({"path": virtual, "is_dir": item.is_dir(), "size": stat.st_size,
                                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                                "writable": not (virtual == SCRATCH or virtual.startswith(SCRATCH + "/"))})
        elif path in ("/inputs", "/findings") or path.startswith(("/inputs/", "/findings/")):
            aliases = self.aliases()
            live = self.aliases(live_only=True)
            paths = (set(state_files) - set(aliases)) | set(live)
            children: dict[str, dict] = {}
            for key in sorted(paths):
                if key not in live and state_files.get(key) is None:
                    continue
                if not key.startswith(path + "/"):
                    continue
                tail = key[len(path) + 1:]
                name = tail.split("/", 1)[0]
                if not hidden and name.startswith("."):
                    continue
                virtual = f"{path}/{name}"
                is_dir = "/" in tail
                children[virtual] = {"path": virtual, "is_dir": is_dir, "writable": False,
                                     "modified_at": (state_files.get(key) or {}).get("modified_at"),
                                     "size": 0 if is_dir else (self.path(live[key]).stat().st_size if key in live
                                                              else len(file_data_to_string(state_files[key]).encode("utf-8")))}
            entries = list(children.values())
        else:
            raise PermissionError("This path is not part of the conversation filesystem")
        entries.sort(key=lambda e: (not e["is_dir"], e["path"]))
        for entry in entries:
            entry.setdefault("modified_at", None)
        return {"path": path, "entries": entries[offset:offset + limit],
                "next_offset": offset + limit if len(entries) > offset + limit else None}

    def preview(self, path: str, state_files: dict[str, Any]) -> dict[str, Any]:
        path = str(assert_safe_path(path))
        content = self.read(path, state_files, limit=MAX_EDIT_BYTES + 1)
        resolved = self.aliases().get(path, path)
        if resolved.startswith("/workspace/"):
            stat = self.path(resolved).stat()
            size = stat.st_size
            modified_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
        else:
            size = len(self.read(path, state_files))
            modified_at = state_files[path].get("modified_at")
        try:
            text = content.decode("utf-8")
            text_file = "\x00" not in text
        except UnicodeDecodeError:
            # A bounded read may end in the middle of a UTF-8 code point.
            try:
                import codecs
                text = codecs.getincrementaldecoder("utf-8")().decode(content, final=size <= MAX_EDIT_BYTES)
                text_file = "\x00" not in text
            except UnicodeDecodeError:
                text, text_file = "", False
        writable = path.startswith("/workspace/") and not path.startswith(SCRATCH + "/")
        return {"path": path, "size": size, "modified_at": modified_at, "writable": writable,
                "text": text[:PREVIEW_BYTES] if size > MAX_EDIT_BYTES else text,
                "is_text": text_file, "truncated": size > MAX_EDIT_BYTES,
                "editable": writable and text_file and size <= MAX_EDIT_BYTES,
                "revision": hashlib.sha256(content).hexdigest() if size <= MAX_EDIT_BYTES else None}

    def save(self, path: str, text: str, revision: str) -> str:
        path = str(assert_safe_path(path))
        if not path.startswith("/workspace/") or path == SCRATCH or path.startswith(SCRATCH + "/"):
            raise PermissionError("This path is read-only")
        content = text.encode("utf-8")
        if len(content) > MAX_EDIT_BYTES or "\x00" in text:
            raise ValueError("Editing requires UTF-8 text of at most 1 MiB")
        original = self.read(path, {}, limit=MAX_EDIT_BYTES + 1)
        if len(original) > MAX_EDIT_BYTES or b"\x00" in original:
            raise ValueError("This file is not editable text")
        original.decode("utf-8")
        return self.store.put(self.workspace_id, path, content, revision=revision)


_filesystem: ConversationFilesystem | None = None


def configure_filesystem(filesystem: ConversationFilesystem) -> None:
    global _filesystem
    _filesystem = filesystem


def get_filesystem() -> ConversationFilesystem | None:
    return _filesystem
