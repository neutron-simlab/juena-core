"""Deep Agents adapters over the same files served by the browser API."""

from __future__ import annotations

__all__ = ['ConversationBackend']

import fnmatch
from pathlib import PurePosixPath

from deepagents.backends import StateBackend
from deepagents.backends.protocol import (
    BackendProtocol, EditResult, FileDownloadResponse, FileUploadResponse,
    GlobResult, GrepResult, LsResult, ReadResult, WriteResult,
)
from deepagents.backends.utils import create_file_data, perform_string_replacement, slice_read_response
from langgraph.runtime import get_runtime

from juena_core.filesystem.service import get_filesystem
from juena_core.runtime_context import _context_value


class ConversationBackend(BackendProtocol):
    """Route-local paths; identity always comes from trusted runtime context."""

    def __init__(self, *, root: str = "/workspace", writable: bool = False,
                 directories_only: bool = False) -> None:
        self.root, self.writable, self.directories_only = root, writable, directories_only

    def _resolve(self):
        runtime = get_runtime()
        context = runtime.context
        filesystem = get_filesystem()
        if filesystem is None:
            raise RuntimeError("Conversation filesystem is not configured")
        return filesystem.for_thread(_context_value(context, "user_id"),
                                     _context_value(context, "thread_id")), runtime

    def _state_files(self):
        return StateBackend()._read_files() if self.root == "/inputs" else {}

    def _full(self, path: str) -> str:
        return self.root + (path if path.startswith("/") else "/" + path)

    def _local(self, entry: dict) -> dict:
        return {k: (v[len(self.root):] or "/") if k == "path" else v
                for k, v in entry.items() if k in {"path", "size", "is_dir", "modified_at", "line", "text"}}

    def ls(self, path: str) -> LsResult:
        try:
            files, runtime = self._resolve()
            with files.guard(shared=True):
                entries, offset = [], 0
                while True:
                    page = files.list(self._full(path), self._state_files(), hidden=True, offset=offset)
                    entries.extend(self._local(e) for e in page["entries"])
                    if page["next_offset"] is None or len(entries) >= 1000:
                        break
                    offset = page["next_offset"]
                return LsResult(entries=entries)
        except (OSError, ValueError, RuntimeError) as exc:
            return LsResult(error=str(exc))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        if self.directories_only:
            return ReadResult(error="Delegate file inspection to a specialist; you can list workspace directories.")
        try:
            files, runtime = self._resolve()
            with files.guard(shared=True):
                content = files.read(self._full(file_path), self._state_files(), limit=10 * 1024 * 1024 + 1)
                if len(content) > 10 * 1024 * 1024:
                    raise ValueError("File is too large for text tools; use sandboxed processing")
                text = content.decode("utf-8")
                return slice_read_response(create_file_data(text), offset, limit)
        except (OSError, ValueError, RuntimeError) as exc:
            return ReadResult(error=str(exc))

    def _publish(self, runtime, path: str, content: bytes) -> None:
        if not path.startswith("/workspace/outputs/"):
            return
        from juena_core.artifacts import get_artifact_store
        context = runtime.context
        store = get_artifact_store()
        user_id, thread_id = _context_value(context, "user_id"), _context_value(context, "thread_id")
        name = path.removeprefix("/workspace/outputs/")
        try:
            store.register_artifact(user_id=user_id, thread_id=thread_id,
                                    run_id=_context_value(context, "run_id"),
                                    filename=PurePosixPath(path).name, content=content, workspace_path=path)
        except (ValueError, OSError) as exc:
            store.note_undelivered(user_id, thread_id, name, str(exc))

    def write(self, file_path: str, content: str) -> WriteResult:
        if not self.writable:
            return WriteResult(error="This filesystem route is read-only")
        try:
            files, runtime = self._resolve()
            path, data = self._full(file_path), content.encode("utf-8")
            with files.guard():
                files.ensure()
                files.store.put(files.workspace_id, path, data, create=True)
            self._publish(runtime, path, data)
            return WriteResult(path=file_path)
        except (OSError, ValueError, RuntimeError) as exc:
            return WriteResult(error=str(exc))

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        if not self.writable:
            return EditResult(error="This filesystem route is read-only")
        try:
            files, runtime = self._resolve()
            path = self._full(file_path)
            with files.guard():
                # Match Deep Agents' text backend's universal newline handling.
                content = files.read(path, {}).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
                result = perform_string_replacement(content, old_string, new_string, replace_all)
                if isinstance(result, str):
                    return EditResult(error=result)
                updated, occurrences = result
                files.store.put(files.workspace_id, path, updated.encode("utf-8"))
            self._publish(runtime, path, updated.encode("utf-8"))
            return EditResult(path=file_path, occurrences=occurrences)
        except (OSError, ValueError, RuntimeError) as exc:
            return EditResult(error=str(exc))

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        try:
            pending, matches, visited = [path or "/"], [], 0
            while pending and visited < 10000:
                result = self.ls(pending.pop())
                if result.error:
                    return GlobResult(error=result.error, matches=matches)
                for entry in result.entries or []:
                    visited += 1
                    if entry.get("is_dir"):
                        pending.append(entry["path"])
                    elif fnmatch.fnmatch(entry["path"], pattern) or PurePosixPath(entry["path"]).match(pattern):
                        matches.append(entry)
            return GlobResult(matches=matches, truncated=bool(pending))
        except (OSError, ValueError, RuntimeError) as exc:
            return GlobResult(error=str(exc))

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        if self.directories_only:
            return GrepResult(error="Delegate file inspection to a specialist")
        try:
            files, _runtime = self._resolve()
            full = self._full(path or "/")
            with files.guard(shared=True):
                is_file = (full in self._state_files() or full in files.aliases()
                           if self.root == "/inputs" else files.path(full).is_file())
            if is_file:
                selected = not glob or fnmatch.fnmatch(path, glob) or PurePosixPath(path).match(glob)
                listing = GlobResult(matches=[{"path": path}] if selected else [])
            else:
                listing = self.glob(glob or "*", path)
        except (OSError, ValueError, RuntimeError) as exc:
            return GrepResult(error=str(exc))
        matches = []
        for entry in listing.matches or []:
            read = self.read(entry["path"], limit=10000)
            if read.error or not read.file_data:
                continue
            for line, text in enumerate(read.file_data["content"].splitlines(), 1):
                if pattern in text:
                    matches.append({"path": entry["path"], "line": line, "text": text})
                    if len(matches) >= 200:
                        return GrepResult(matches=matches, truncated=True)
        return GrepResult(matches=matches, error=listing.error, truncated=listing.truncated)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        results = []
        for path, content in files:
            try:
                result = self.write(path, content.decode("utf-8"))
                results.append(FileUploadResponse(path=path, error=result.error))
            except UnicodeDecodeError:
                results.append(FileUploadResponse(path=path, error="Text writes require UTF-8"))
        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        if self.directories_only:
            return [FileDownloadResponse(path=p, error="permission_denied") for p in paths]
        results = []
        for path in paths:
            try:
                files, runtime = self._resolve()
                with files.guard(shared=True):
                    content = files.read(self._full(path), self._state_files(), limit=10 * 1024 * 1024 + 1)
                    if len(content) > 10 * 1024 * 1024:
                        raise ValueError("Download exceeds the tool transfer limit")
                results.append(FileDownloadResponse(path=path, content=content))
            except (OSError, ValueError, RuntimeError) as exc:
                results.append(FileDownloadResponse(path=path, error=str(exc)))
        return results
