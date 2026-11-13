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
    """Every conversation's files, kept in one workspace store."""

    def __init__(self, store: WorkspaceStore, *, identity_secret: str,
                 active_workspaces: Callable[[], set[str]] | None = None) -> None:
        """Initialize the filesystem.

        Args:
            store: Where the workspaces live on disk.
            identity_secret: Secret key that turns user and thread ids into
                workspace ids; at least 32 characters.
            active_workspaces: Returns the ids of workspaces with a queued or
                running execution; ``None`` means there are never any.

        Raises:
            ValueError: ``identity_secret`` is shorter than 32 characters.
        """
        if len(identity_secret) < 32:
            raise ValueError("Filesystem identity secret must contain at least 32 characters")
        self.store = store
        self.identity_secret = identity_secret
        self.active_workspaces = active_workspaces or set

    def assert_idle(self, workspace_id: str) -> None:
        """Raise when the workspace has a queued or running execution.

        Args:
            workspace_id: The workspace to check.

        Raises:
            WorkspaceBusy: An execution is queued or running.
        """
        # Durable queues can outlive the process that held the operation lock.
        if workspace_id in self.active_workspaces():
            raise WorkspaceBusy("Conversation has a queued or running execution")

    @asynccontextmanager
    async def lifespan(self):
        """Remove expired scratch files once an hour while the context is open.

        Yields:
            Nothing; the cleanup task stops when the context exits.
        """
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
        """Return one conversation's files.

        Args:
            user_id: The conversation's owner.
            thread_id: The conversation.

        Returns:
            The conversation's files.

        Raises:
            ValueError: Either id is empty.
        """
        if not user_id or not thread_id:
            raise ValueError("Authenticated user and conversation are required")
        return ThreadFiles(self.store, workspace_id_for(user_id, thread_id, self.identity_secret),
                           tenant_id_for(user_id, self.identity_secret))


class ThreadFiles:
    """One conversation's files: its workspace on disk plus files held in graph state."""

    def __init__(self, store: WorkspaceStore, workspace_id: str, tenant_id: str) -> None:
        """Initialize the conversation's files.

        Args:
            store: Where the workspace lives on disk.
            workspace_id: The conversation's workspace id.
            tenant_id: The owner's id, checked against the workspace's owner
                marker.
        """
        self.store, self.workspace_id, self.tenant_id = store, workspace_id, tenant_id

    def ensure(self) -> None:
        """Create the workspace directories if missing, and check the owner.

        Raises:
            WorkspaceDeleted: The conversation's workspace was deleted.
            WorkspaceOwnershipError: The workspace belongs to someone else.
        """
        self.store.ensure(self.workspace_id, self.tenant_id)

    def guard(self, **kwargs):
        """Lock this workspace, as :meth:`WorkspaceStore.guard` does.

        Args:
            **kwargs: Passed to :meth:`WorkspaceStore.guard`.

        Returns:
            A context manager that holds the lock.
        """
        return self.store.guard(self.workspace_id, **kwargs)

    def aguard(self, **kwargs):
        """Lock this workspace without blocking the event loop.

        Args:
            **kwargs: Passed to :meth:`WorkspaceStore.aguard`.

        Returns:
            An async context manager that holds the lock.
        """
        return self.store.aguard(self.workspace_id, **kwargs)

    async def acquire(self, **kwargs):
        """Take this workspace's lock and return it held.

        Args:
            **kwargs: Passed to :meth:`WorkspaceStore.acquire`.

        Returns:
            The held lock; the caller must exit it.
        """
        return await self.store.acquire(self.workspace_id, **kwargs)

    def path(self, path: str, *, write: bool = False) -> Path:
        """Resolve a conversation path to its file on disk.

        Args:
            path: A path under ``/workspace``, or under ``/inputs`` for reading.
            write: Whether the path will be written; ``/inputs`` is refused then.

        Returns:
            The file's location on disk.

        Raises:
            PermissionError: The path is outside what may be accessed.
            ValueError: The path is unsafe, or goes through a link or a
                special file.
        """
        return self.store.resolve_path(self.store.workspace_dir(self.workspace_id),
                                       self.store.inputs_dir(self.workspace_id), path, write=write)

    def aliases(self, *, live_only: bool = False) -> dict[str, str]:
        """Return where each upload held in graph state was copied in the workspace.

        Args:
            live_only: Whether to keep only the copies that still exist.

        Returns:
            Each ``/inputs/uploads/`` path mapped to its ``/workspace/uploads/``
            copy.
        """
        path = self.store.workspace_root(self.workspace_id) / ".uploads.json"
        aliases = json.loads(path.read_text()) if path.exists() else {}
        if live_only:
            return {old: new for old, new in aliases.items() if self.path(new).is_file()}
        return aliases

    def needs_migration(self, files: dict[str, Any]) -> bool:
        """Whether :meth:`migrate` has work to do.

        Args:
            files: The graph's ``files`` channel.

        Returns:
            Whether the workspace is missing or an upload in ``files`` has
            not been copied yet.
        """
        if not self.store.workspace_dir(self.workspace_id).is_dir():
            return True
        aliases = self.aliases()
        return any(path.startswith("/inputs/uploads/") and data is not None and path not in aliases
                   for path, data in files.items())

    def migrate(self, files: dict[str, Any]) -> dict[str, str]:
        """Copy uploads held in graph state into ``/workspace/uploads/``.

        Called under the files lock. A saved mapping survives subsequent user
        edits.

        Args:
            files: The graph's ``files`` channel.

        Returns:
            The full map from upload path to workspace copy.
        """
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
        """Save one upload, as :meth:`add_uploads` does.

        Args:
            name: The file name.
            content: The file's bytes.

        Returns:
            The upload's path.
        """
        return self.add_uploads([(name, content)])[0]

    def add_uploads(self, uploads: list[tuple[str, bytes]]) -> list[str]:
        """Publish a validated batch under the files guard, rolling back failed writes.

        Args:
            uploads: ``(name, content)`` pairs.

        Returns:
            Each upload's path under ``/workspace/uploads/``; a taken name gets
            a ``_2``, ``_3``, ... suffix.
        """
        created: list[str] = []
        try:
            for name, content in uploads:
                candidate = f"/workspace/uploads/{name}"
                index = 2
                while self.path(candidate).exists():
                    candidate = f"/workspace/uploads/{Path(name).stem}_{index}{Path(name).suffix}"
                    index += 1
                created.append(candidate)
                self.store.put(self.workspace_id, candidate, content, create=True)
        except BaseException:
            for path in created:
                self.path(path).unlink(missing_ok=True)
            self.store.invalidate_usage(self.workspace_id)
            raise
        return created

    def read(self, path: str, state_files: dict[str, Any], *, limit: int | None = None) -> bytes:
        """Read a conversation file's bytes.

        ``/inputs/`` and ``/findings/`` come from graph state, everything else
        from the workspace; an upload's old path reads its workspace copy.

        Args:
            path: The file.
            state_files: The graph's ``files`` channel.
            limit: Most bytes to read; ``None`` reads the whole file.

        Returns:
            The file's bytes.

        Raises:
            FileNotFoundError: There is no such file.
            WorkspaceOwnershipError: The workspace belongs to someone else.
            ValueError: The path is unsafe.
        """
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
        """List one page of a directory, folders first.

        Args:
            path: ``/``, or a directory under ``/workspace``, ``/inputs`` or
                ``/findings``.
            state_files: The graph's ``files`` channel.
            hidden: Whether to include names that start with a dot.
            offset: Index of the first entry to return.
            limit: Most entries to return.

        Returns:
            ``path``, the page of ``entries``, and ``next_offset`` (``None`` on
            the last page).

        Raises:
            FileNotFoundError: The workspace directory does not exist.
            PermissionError: The path is outside the conversation filesystem.
        """
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
        """Describe a file for the file browser, with its text when it has some.

        Files over ``MAX_EDIT_BYTES`` show only their first ``PREVIEW_BYTES``
        of text and cannot be edited.

        Args:
            path: The file.
            state_files: The graph's ``files`` channel.

        Returns:
            Size, modification time, text, and whether the file is text,
            truncated, writable and editable, plus the ``revision`` that
            :meth:`save` needs.
        """
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
        """Replace an editable workspace file with new text.

        Args:
            path: A file under ``/workspace``, outside the scratch area.
            text: The new content, at most 1 MiB.
            revision: The revision the edit started from.

        Returns:
            The file's new revision.

        Raises:
            PermissionError: The path is read-only.
            ValueError: The new or the old content is not editable text.
            RevisionConflict: The file changed since ``revision``.
        """
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
    """Install the process-wide conversation filesystem.

    Args:
        filesystem: The filesystem to use.
    """
    global _filesystem
    _filesystem = filesystem


def get_filesystem() -> ConversationFilesystem | None:
    """Return the process-wide conversation filesystem.

    Returns:
        The installed filesystem, or ``None`` when the application has none.
    """
    return _filesystem
