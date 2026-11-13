"""Deep Agents backends that delegate execution to the sandbox worker."""

from __future__ import annotations

from pathlib import PurePosixPath
from contextlib import nullcontext
from typing import Any

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.sandbox import (
    BaseSandbox,
    _build_glob_cmd,
    _build_grep_cmd,
    _parse_glob_output,
    _parse_grep_output,
)
from langgraph.config import get_stream_writer

from juena_core.artifacts import get_artifact_store, undelivered_note
from juena_core.sandbox.config import sandbox_settings
from juena_core.sandbox.evidence import record_sandbox_execution
from juena_core.sandbox.jobs import SandboxJobs
from juena_core.sandbox.policy import is_redundant_artifact_export
from juena_core.schema.interrupts import ArtifactRef
from juena_core.sandbox.workspace import (
    WorkspaceFileResult,
    WorkspaceStore,
    assert_safe_path,
)

__all__ = ["PodmanSandboxBackend", "ReadOnlySandboxInputsBackend", "STATUS_LABELS"]

# Every status the backend can report, including the two the sandbox returns
# without ever writing a row.
STATUS_LABELS = {
    "queued": "Sandbox command queued",
    "running": "Sandbox command running",
    "completed": "Sandbox command completed",
    "timed_out": "Sandbox command timed out",
    "failed": "Sandbox command failed",
    "busy": "Sandbox capacity is busy",
    "unavailable": "Sandbox worker unavailable",
}


def _artifact_note(
    attached: list[ArtifactRef], dropped: list[tuple[str, str]], *, succeeded: bool
) -> str:
    """Describe what reached the user and what did not.

    A file the sandbox wrote but the application refused is the one thing the
    model must never guess about: with no note it sees a clean exit code and
    reports a figure the user cannot open. Only a successful command is told
    to finish; after a failure the model may still need to fix and rerun.
    """
    blocks: list[str] = []
    if attached:
        lines = "\n".join(
            f"- {ref.artifact_id} | {ref.filename} | {ref.mime_type} | {ref.caption}"
            for ref in attached
        )
        if succeeded:
            advice = (
                "\nDo not run another command to inspect, encode, print, or "
                "verify these files. Finish the report now."
            )
        else:
            advice = (
                "\nThe command failed, but these files were collected and will reach "
                "the user as they are. If the task still needs a working result, fix "
                "the problem and run again; files written again replace these. Do not "
                "run a command only to inspect, encode, or print them."
            )
        blocks.append(
            "Artifacts collected automatically by the application:\n" + lines + advice
        )
    if dropped:
        blocks.append(undelivered_note(dropped))
    return "\n\n".join(blocks)


class PodmanSandboxBackend(BaseSandbox):
    """Tenant-bound backend whose commands run only through the host worker."""

    def __init__(
        self,
        *,
        jobs: SandboxJobs,
        workspaces: WorkspaceStore,
        tenant_id: str,
        workspace_id: str,
        user_id: str,
        thread_id: str,
        run_id: str | None,
        control: Any = None,
    ) -> None:
        """Initialize a backend for one conversation's workspace.

        Args:
            jobs: The job queue that commands are submitted through.
            workspaces: The store that holds the workspace files.
            tenant_id: The hashed id of the user who owns the workspace.
            workspace_id: The hashed id of the conversation's workspace.
            user_id: The signed-in user, for artifacts and the audit log.
            thread_id: The conversation, for artifacts and the audit log.
            run_id: The run that jobs and artifacts are recorded under, if any.
            control: The run's stop control, or None. Each command runs as one
                of its operations, and the run is checked once the job ends.
        """
        self._jobs = jobs
        self._workspaces = workspaces
        self._tenant_id = tenant_id
        self._workspace_id = workspace_id
        self._user_id = user_id
        self._thread_id = thread_id
        self._run_id = run_id
        self._control = control

    @property
    def id(self) -> str:
        """The workspace id."""
        return self._workspace_id

    @staticmethod
    def _status(status: str, label: str, **extra: Any) -> None:
        try:
            get_stream_writer()(
                {"type": "sandbox_status", "status": status, "label": label, **extra}
            )
        except (KeyError, RuntimeError):
            # Backend unit tests and non-streaming invocations have no writer.
            return

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a command in the sandbox and collect what it wrote to ``/workspace/outputs``.

        A command that only prints a stored output as base64 is not run. An
        error while running comes back as an ``exit_code`` of None with the
        reason in the output, so the chat keeps going.

        Args:
            command: The shell command to run.
            timeout: Seconds to allow. None, or more than the configured limit,
                means the configured limit.

        Returns:
            The output for the model, with a note on which files reached the
            user, the exit code, and whether the output was cut short.
        """
        if self._control is not None:
            self._control.sandbox_jobs = self._jobs
        with self._control.operation() if self._control is not None else nullcontext():
            return self._execute(command, timeout=timeout)

    def _execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        config = sandbox_settings()
        effective_timeout = min(
            timeout or config.execution_timeout_seconds,
            config.execution_timeout_seconds,
        )
        store = get_artifact_store()
        if is_redundant_artifact_export(command):
            store.audit(
                {
                    "event": "sandbox_execution_skipped",
                    "user_id": self._user_id,
                    "thread_id": self._thread_id,
                    "run_id": self._run_id,
                    "command": command,
                    "reason": "artifact_already_collected",
                }
            )
            record_sandbox_execution(
                command=command,
                status="skipped",
                exit_code=0,
            )
            return ExecuteResponse(
                output=(
                    "The application already collects files under /workspace/outputs "
                    "and displays PNG artifacts automatically. Do not encode or print "
                    "the file; finish the report using the collected artifact metadata."
                ),
                exit_code=0,
            )
        try:
            result = self._jobs.execute(
                tenant_id=self._tenant_id,
                workspace_id=self._workspace_id,
                command=command,
                timeout_seconds=effective_timeout,
                max_output_bytes=config.max_output_bytes,
                run_id=self._run_id,
                status_callback=lambda status: self._status(
                    status,
                    STATUS_LABELS.get(status, "Sandbox command running"),
                ),
            )
            if self._control is not None:
                self._control.check()
            attached: list[ArtifactRef] = []
            dropped: list[tuple[str, str]] = []
            paths = [f"/workspace/outputs/{name}" for name in result.artifact_names]
            collected = self._collect(paths)
            # _collect returns nothing at all when the workspace guard is busy, so
            # a file the worker really wrote would otherwise vanish without a word.
            returned = {item.path for item in collected}
            dropped.extend(
                (
                    PurePosixPath(path).name,
                    "The workspace was busy and the file could not be read",
                )
                for path in paths
                if path not in returned
            )
            for artifact in collected:
                name = PurePosixPath(artifact.path).name
                if artifact.error or artifact.content is None:
                    dropped.append((name, artifact.error or "The file could not be read"))
                    continue
                try:
                    attached.append(
                        store.register_artifact(
                            user_id=self._user_id,
                            thread_id=self._thread_id,
                            run_id=self._run_id,
                            filename=name,
                            content=artifact.content,
                            workspace_path=artifact.path,
                        )
                    )
                except (OSError, ValueError) as exc:
                    dropped.append((name, str(exc)))
            store.audit(
                {
                    "event": "sandbox_execution",
                    "user_id": self._user_id,
                    "thread_id": self._thread_id,
                    "run_id": self._run_id,
                    "job_id": str(result.job_id),
                    "command": command,
                    "status": result.status,
                    "exit_code": result.exit_code,
                    "artifact_ids": [ref.artifact_id for ref in attached],
                    "dropped": [name for name, _ in dropped],
                }
            )
            self._status(
                result.status,
                STATUS_LABELS.get(result.status, "Sandbox command failed"),
                job_id=str(result.job_id),
                exit_code=result.exit_code,
            )
            record_sandbox_execution(
                command=command,
                status=result.status,
                exit_code=result.exit_code,
                truncated=result.truncated,
                artifact_ids=[ref.artifact_id for ref in attached],
                artifact_filenames=[ref.filename for ref in attached],
                dropped=dropped,
            )
            model_output = result.output.rstrip()
            note = _artifact_note(
                attached,
                dropped,
                succeeded=result.status == "completed" and result.exit_code == 0,
            )
            if note:
                model_output = "\n\n".join(part for part in (model_output, note) if part)
            return ExecuteResponse(
                output=model_output,
                exit_code=result.exit_code,
                truncated=result.truncated,
            )
        except Exception as exc:  # noqa: BLE001 - keep chat alive on sandbox outages
            store.audit(
                {
                    "event": "sandbox_execution",
                    "user_id": self._user_id,
                    "thread_id": self._thread_id,
                    "run_id": self._run_id,
                    "command": command,
                    "status": "worker_error",
                }
            )
            self._status("failed", "Sandbox worker unavailable")
            record_sandbox_execution(
                command=command,
                status="worker_error",
                exit_code=None,
            )
            return ExecuteResponse(output=f"Sandbox unavailable: {exc}", exit_code=None)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching a glob pattern, searching inside the sandbox.

        Args:
            pattern: The glob pattern.
            path: The directory to search. None means ``/workspace``; ``/`` is refused.

        Returns:
            The matching files, or an error.
        """
        search_path = path or "/workspace"
        if PurePosixPath(search_path) == PurePosixPath("/"):
            return GlobResult(error="Recursive searches from / are not allowed")
        result = self.execute(_build_glob_cmd(pattern, search_path), timeout=15)
        if result.exit_code not in (0, None):
            return GlobResult(error=result.output)
        return _parse_glob_output(result.output, search_path)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search file contents by running grep inside the sandbox.

        Args:
            pattern: The pattern to search for.
            path: The directory to search. None means ``/workspace``; ``/`` is refused.
            glob: Search only files matching this glob, if given.

        Returns:
            The matching lines, or an error.
        """
        search_path = path or "/workspace"
        if PurePosixPath(search_path) == PurePosixPath("/"):
            return GrepResult(error="Recursive searches from / are not allowed")
        result = self.execute(_build_grep_cmd(pattern, search_path, glob), timeout=15)
        if result.exit_code not in (0, 1, None):
            return GrepResult(error=result.output)
        return _parse_grep_output(result, search_path)

    def _collect(self, paths: list[str]) -> list[WorkspaceFileResult]:
        """Read finished outputs under the same guard as explicit file access."""

        if not paths:
            return []
        with self._jobs.workspace_guard(self._workspace_id) as available:
            if not available:
                return []
            return self._workspaces.download(self._workspace_id, self._tenant_id, paths)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Write files into the workspace.

        Files under ``/workspace/outputs/`` are also registered as artifacts for
        the user. If that fails, the write still counts as done and the user is
        told through the undelivered-files note instead.

        Args:
            files: ``(path, content)`` pairs.

        Returns:
            One response per file, in order, with an error when the write failed
            or the workspace was busy.
        """
        if not files:
            return []
        with self._jobs.workspace_guard(self._workspace_id) as available:
            if not available:
                return [
                    FileUploadResponse(path=path, error="workspace busy")
                    for path, _content in files
                ]
            workspace_results = self._workspaces.upload(
                self._workspace_id,
                self._tenant_id,
                files,
            )
        responses = [
            FileUploadResponse(
                path=path,
                error=(
                    workspace_results[index].error
                    if index < len(workspace_results)
                    else "missing workspace response"
                ),
            )
            for index, (path, _content) in enumerate(files)
        ]
        store = None
        for (path, content), response in zip(files, responses, strict=False):
            if response.error or not path.startswith("/workspace/outputs/"):
                continue
            if store is None:
                store = get_artifact_store()
            try:
                store.register_artifact(
                    user_id=self._user_id,
                    thread_id=self._thread_id,
                    run_id=self._run_id,
                    filename=PurePosixPath(path).name,
                    content=content,
                    workspace_path=path,
                )
            except (ValueError, OSError) as exc:
                # Not reported through FileUploadResponse.error: the write itself
                # succeeded and the bytes are on disk, so `write_file` would say
                # "Failed to write file" about a file that exists -- and a retry
                # then collides with it, because write() refuses an existing
                # path. The notice waits for a channel that can carry it.
                reason = str(exc)
                filename = PurePosixPath(path).name
                store.audit(
                    {
                        "event": "artifact_dropped",
                        "user_id": self._user_id,
                        "thread_id": self._thread_id,
                        "run_id": self._run_id,
                        "filename": filename,
                        "reason": reason,
                    }
                )
                store.note_undelivered(self._user_id, self._thread_id, filename, reason)
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Read files from the workspace.

        Args:
            paths: The files to read.

        Returns:
            One response per path, in order, with the content or an error such
            as ``workspace busy``.
        """
        if not paths:
            return []
        with self._jobs.workspace_guard(self._workspace_id) as available:
            if not available:
                return [FileDownloadResponse(path=path, error="workspace busy") for path in paths]
            workspace_results = self._workspaces.download(
                self._workspace_id,
                self._tenant_id,
                paths,
            )
        responses: list[FileDownloadResponse] = []
        for index, path in enumerate(paths):
            if index >= len(workspace_results):
                responses.append(FileDownloadResponse(path=path, error="missing workspace response"))
                continue
            item = workspace_results[index]
            responses.append(
                FileDownloadResponse(path=path, content=item.content, error=item.error)
            )
        return responses


class ReadOnlySandboxInputsBackend(BackendProtocol):
    """Route-local `/inputs` adapter over the sandbox's read-only input mount."""

    def __init__(self, sandbox: SandboxBackendProtocol) -> None:
        """Initialize the adapter.

        Args:
            sandbox: The sandbox backend whose ``/inputs`` mount is read.
        """
        self._sandbox = sandbox

    @staticmethod
    def _full(path: str | None) -> str:
        """Map a route-local path to the sandbox's ``/inputs`` path, refusing unsafe ones."""
        local = path or "/"
        assert_safe_path(local, "Invalid /inputs path")
        return "/inputs" if local == "/" else f"/inputs{local}"

    @staticmethod
    def _local(path: str) -> str:
        """Map a sandbox ``/inputs`` path back to a route-local one."""
        if path == "/inputs":
            return "/"
        if path.startswith("/inputs/"):
            return path[len("/inputs") :]
        return path

    def ls(self, path: str) -> LsResult:
        """List a directory under ``/inputs``.

        Args:
            path: The directory, as a path inside ``/inputs``.

        Returns:
            The entries with route-local paths, or an error.
        """
        try:
            result = self._sandbox.ls(self._full(path))
        except ValueError as exc:
            return LsResult(error=str(exc))
        entries = [
            FileInfo(**{**item, "path": self._local(item["path"])})
            for item in (result.entries or [])
        ]
        return LsResult(error=result.error, entries=entries)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a file under ``/inputs``.

        Args:
            file_path: The file, as a path inside ``/inputs``.
            offset: Line to start reading from, counting from 0.
            limit: Maximum number of lines to read.

        Returns:
            The file's content, or an error.
        """
        try:
            return self._sandbox.read(self._full(file_path), offset=offset, limit=limit)
        except ValueError as exc:
            return ReadResult(error=str(exc))

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search file contents under ``/inputs``.

        Args:
            pattern: The pattern to search for.
            path: The directory, as a path inside ``/inputs``. None means all of it.
            glob: Search only files matching this glob, if given.

        Returns:
            The matches with route-local paths, or an error.
        """
        try:
            result = self._sandbox.grep(pattern, self._full(path), glob)
        except ValueError as exc:
            return GrepResult(error=str(exc))
        matches = [
            GrepMatch(**{**item, "path": self._local(item["path"])})
            for item in (result.matches or [])
        ]
        return GrepResult(error=result.error, matches=matches)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files under ``/inputs`` matching a glob pattern.

        Args:
            pattern: The glob pattern.
            path: The directory, as a path inside ``/inputs``. None means all of it.

        Returns:
            The matching files with route-local paths, or an error.
        """
        try:
            result = self._sandbox.glob(pattern, self._full(path))
        except ValueError as exc:
            return GlobResult(error=str(exc))
        matches = [
            FileInfo(**{**item, "path": self._local(item["path"])})
            for item in (result.matches or [])
        ]
        return GlobResult(error=result.error, matches=matches)

    @staticmethod
    def write(file_path: str, content: str) -> WriteResult:
        """Refuse a write, because staged inputs are read-only.

        Args:
            file_path: The path the caller asked to write.
            content: Ignored.

        Returns:
            An error naming the path.
        """
        del content
        return WriteResult(error=f"Cannot write to read-only staged input path '{file_path}'.")

    @staticmethod
    def edit(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Refuse an edit, because staged inputs are read-only.

        Args:
            file_path: The path the caller asked to edit.
            old_string: Ignored.
            new_string: Ignored.
            replace_all: Ignored.

        Returns:
            An error naming the path.
        """
        del old_string, new_string, replace_all
        return EditResult(error=f"Cannot edit read-only staged input path '{file_path}'.")

    @staticmethod
    def upload_files(files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Refuse every upload, because staged inputs are read-only.

        Args:
            files: ``(path, content)`` pairs.

        Returns:
            A ``permission_denied`` error for each file.
        """
        return [
            FileUploadResponse(path=path, error="permission_denied") for path, _content in files
        ]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Read files under ``/inputs``.

        Args:
            paths: The files, as paths inside ``/inputs``.

        Returns:
            One response per path. If any path is unsafe, every response
            carries that error.
        """
        full_paths: list[str] = []
        for path in paths:
            try:
                full_paths.append(self._full(path))
            except ValueError as exc:
                return [FileDownloadResponse(path=item, error=str(exc)) for item in paths]
        results = self._sandbox.download_files(full_paths)
        return [
            FileDownloadResponse(path=path, content=result.content, error=result.error)
            for path, result in zip(paths, results, strict=False)
        ]
