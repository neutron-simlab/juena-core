"""Postgres-backed sandbox admission, claiming, and results."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import monotonic, sleep
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from juena_core.sandbox.constants import ACTIVE_STATUSES, TERMINAL_STATUSES, JobStatus
from juena_core.filesystem.storage import tenant_id_for, workspace_id_for

__all__ = [
    "CLAIM_DEADLINE_SECONDS",
    "CLAIM_POLL_SECONDS",
    "RESULT_POLL_SECONDS",
    "SandboxJobRecord",
    "SandboxExecutionResult",
    "SandboxJobs",
    "database_url",
    "tenant_id_for",
    "workspace_id_for",
    "hold_worker_lock",
]

# The worker holds this session lock for its lifetime; the API only observes it.
WORKER_LOCK_CLASSID = 0x4A55
WORKER_LOCK_OBJID = 0x454E

# Queued rows are short-lived claims, not a waiting queue.
CLAIM_DEADLINE_SECONDS = 15
CLAIM_POLL_SECONDS = 0.25
RESULT_POLL_SECONDS = 0.25

_ADMISSION_LOCK = 0x4A55454E41534E44
_DATABASE_URL_PREFIXES = ("postgresql+psycopg://", "postgresql://", "postgres://")
_COLUMNS = (
    "id, tenant_id, workspace_id, command, timeout_seconds, max_output_bytes, "
    "status, output, exit_code, truncated, artifact_names, created_at, "
    "started_at, finished_at, run_id, cancel_requested"
)


def database_url(url: str) -> str:
    """Strip the SQLAlchemy driver suffix psycopg does not understand.

    Args:
        url: A Postgres URL, with or without ``+psycopg``.

    Returns:
        The same URL starting with ``postgresql://``.

    Raises:
        ValueError: The URL is not a Postgres one.
    """
    for prefix in _DATABASE_URL_PREFIXES:
        if url.startswith(prefix):
            return f"postgresql://{url.removeprefix(prefix)}"
    raise ValueError("DATABASE_URL must use postgres:// or postgresql://")


def _workspace_lock_key(workspace_id: str) -> int:
    digest = hashlib.sha256(f"workspace:{workspace_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


@dataclass(frozen=True, slots=True)
class SandboxJobRecord:
    """One row of the ``sandbox_jobs`` table.

    Attributes:
        id: The job's id.
        tenant_id: The hashed id of the user who owns the workspace.
        workspace_id: The hashed id of the conversation's workspace.
        command: The shell command.
        timeout_seconds: The longest the command may run.
        max_output_bytes: How much of its output is kept.
        status: ``queued``, ``running``, or one of the terminal statuses.
        output: The command's output, or the reason it failed.
        exit_code: The command's exit code, if it has one.
        truncated: Whether the output was cut at ``max_output_bytes``.
        artifact_names: The output files accepted for delivery.
        created_at: When the job was queued.
        started_at: When a worker slot claimed it.
        finished_at: When it reached a terminal status.
        run_id: The run that submitted it, so stopping the run can cancel it.
        cancel_requested: Whether stopping the run asked to cancel it.
    """

    id: UUID
    tenant_id: str
    workspace_id: str
    command: str
    timeout_seconds: int
    max_output_bytes: int
    status: JobStatus
    output: str
    exit_code: int | None
    truncated: bool
    artifact_names: tuple[str, ...]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    run_id: str | None = None
    cancel_requested: bool = False


@dataclass(frozen=True, slots=True)
class SandboxExecutionResult:
    """What a submitted command came back with.

    Attributes:
        status: A job status, or ``busy`` / ``unavailable`` when no job was created.
        output: The command's output, or why it did not run.
        job_id: The job's id, or None when no job was created.
        exit_code: The command's exit code, if it has one.
        truncated: Whether the output was cut at the byte limit.
        artifact_names: The output files accepted for delivery.
    """

    status: str
    output: str
    job_id: UUID | None = None
    exit_code: int | None = None
    truncated: bool = False
    artifact_names: tuple[str, ...] = ()


def _record(row: dict) -> SandboxJobRecord:
    return SandboxJobRecord(
        id=row["id"],
        tenant_id=row["tenant_id"],
        workspace_id=row["workspace_id"],
        command=row["command"],
        timeout_seconds=row["timeout_seconds"],
        max_output_bytes=row["max_output_bytes"],
        status=row["status"],
        output=row["output"],
        exit_code=row["exit_code"],
        truncated=row["truncated"],
        artifact_names=tuple(row["artifact_names"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        run_id=row.get("run_id"),
        cancel_requested=row.get("cancel_requested", False),
    )


def _result(record: SandboxJobRecord) -> SandboxExecutionResult:
    return SandboxExecutionResult(
        status=record.status,
        output=record.output,
        job_id=record.id,
        exit_code=record.exit_code,
        truncated=record.truncated,
        artifact_names=record.artifact_names,
    )


@contextmanager
def hold_worker_lock(url: str) -> Iterator[bool]:
    """Hold the worker session lock on a dedicated connection.

    Args:
        url: The Postgres URL of the job queue.

    Yields:
        Whether the lock was acquired. False means another worker holds it.
    """

    with psycopg.connect(database_url(url), autocommit=True) as connection:
        row = connection.execute(
            "SELECT pg_try_advisory_lock(%s, %s)",
            (WORKER_LOCK_CLASSID, WORKER_LOCK_OBJID),
        ).fetchone()
        yield bool(row and row[0])


class SandboxJobs:
    """Every `sandbox_jobs` operation, for both the application and the worker."""

    def __init__(self, url: str, *, concurrency: int) -> None:
        """Initialize the queue. Call ``open`` before using it.

        Args:
            url: The Postgres URL; a SQLAlchemy driver suffix is accepted.
            concurrency: How many jobs may be active at once.
        """
        self.concurrency = concurrency
        self.pool = ConnectionPool(
            conninfo=database_url(url),
            min_size=0,
            max_size=max(concurrency + 2, 4),
            open=False,
            kwargs={"row_factory": dict_row},
        )

    def open(self) -> None:
        """Open the connection pool and wait until it is ready."""
        self.pool.open(wait=True)

    def close(self) -> None:
        """Close the connection pool."""
        self.pool.close()

    def worker_is_alive(self) -> bool:
        """Read the worker's session lock. objsubid 2 marks the two-int4 form.

        Returns:
            True while a worker holds the lock.
        """

        with self.pool.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND classid = %s "
                "AND objid = %s AND objsubid = 2 AND granted LIMIT 1",
                (WORKER_LOCK_CLASSID, WORKER_LOCK_OBJID),
            ).fetchone()
        return row is not None

    def create(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        command: str,
        timeout_seconds: int,
        max_output_bytes: int,
        run_id: str | None = None,
    ) -> SandboxJobRecord | None:
        """Admit a job without racing capacity or workspace file operations.

        Args:
            tenant_id: The hashed id of the user who owns the workspace.
            workspace_id: The hashed id of the conversation's workspace.
            command: The shell command.
            timeout_seconds: The longest the command may run.
            max_output_bytes: How much of its output to keep.
            run_id: The run submitting it, if any, so stopping the run can cancel it.

        Returns:
            The queued job, or None when capacity or the workspace is busy.
        """

        with self.pool.connection() as connection, connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (_ADMISSION_LOCK,))
            workspace = connection.execute(
                "SELECT pg_try_advisory_xact_lock(%s) AS locked",
                (_workspace_lock_key(workspace_id),),
            ).fetchone()
            if not workspace or not workspace["locked"]:
                return None
            active = connection.execute(
                "SELECT count(*) AS count FROM sandbox_jobs WHERE status = ANY(%s)",
                (list(ACTIVE_STATUSES),),
            ).fetchone()
            if active is None or active["count"] >= self.concurrency:
                return None
            row = connection.execute(
                f"""
                INSERT INTO sandbox_jobs (
                    id, tenant_id, workspace_id, command, timeout_seconds,
                    max_output_bytes, status, output, truncated, artifact_names,
                    created_at, run_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'queued', '', false, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING {_COLUMNS}
                """,
                (
                    uuid4(),
                    tenant_id,
                    workspace_id,
                    command,
                    timeout_seconds,
                    max_output_bytes,
                    Jsonb([]),
                    datetime.now(UTC),
                    run_id,
                ),
            ).fetchone()
        return _record(row) if row is not None else None

    def execute(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        command: str,
        timeout_seconds: int,
        max_output_bytes: int,
        status_callback: Callable[[str], None] | None = None,
        run_id: str | None = None,
    ) -> SandboxExecutionResult:
        """Submit one command and wait for its terminal row.

        Args:
            tenant_id: The hashed id of the user who owns the workspace.
            workspace_id: The hashed id of the conversation's workspace.
            command: The shell command.
            timeout_seconds: The longest the command may run.
            max_output_bytes: How much of its output to keep.
            status_callback: Called with each new status, starting with ``queued``.
            run_id: The run submitting it, if any, so stopping the run can cancel it.

        Returns:
            The finished job's result, ``unavailable`` when no worker is
            running, or ``busy`` when the job was not admitted.
        """

        if not self.worker_is_alive():
            return SandboxExecutionResult("unavailable", "Sandbox worker is unavailable")
        job = self.create(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            command=command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            run_id=run_id,
        )
        if job is None:
            return SandboxExecutionResult("busy", "Sandbox capacity or workspace is busy")
        if status_callback:
            status_callback("queued")

        deadline = monotonic() + CLAIM_DEADLINE_SECONDS + timeout_seconds + 30
        claim_deadline = job.created_at + timedelta(seconds=CLAIM_DEADLINE_SECONDS)
        last_status = "queued"
        while monotonic() < deadline:
            sleep(RESULT_POLL_SECONDS)
            record = self.get(job.id)
            if record is None:
                return SandboxExecutionResult("failed", "Sandbox job disappeared", job.id)
            if record.status in TERMINAL_STATUSES:
                return _result(record)
            if record.status == "queued" and datetime.now(UTC) > claim_deadline:
                failed = self.fail_queued(job.id, "Sandbox worker is unavailable")
                if failed:
                    return _result(failed)
                continue
            if record.status != last_status:
                last_status = record.status
                if status_callback:
                    status_callback(record.status)
        return _result(
            self.finish(job.id, status="failed", output="Sandbox result wait timed out")
            or job
        )

    @contextmanager
    def workspace_guard(self, workspace_id: str) -> Iterator[bool]:
        """Yield False instead of blocking behind an active workspace job.

        Args:
            workspace_id: The workspace to lock.

        Yields:
            True when the workspace is free. It stays locked until the block exits.
        """

        with self.pool.connection() as connection, connection.transaction():
            locked = connection.execute(
                "SELECT pg_try_advisory_xact_lock(%s) AS locked",
                (_workspace_lock_key(workspace_id),),
            ).fetchone()
            active = connection.execute(
                "SELECT 1 FROM sandbox_jobs WHERE workspace_id = %s "
                "AND status = ANY(%s) LIMIT 1",
                (workspace_id, list(ACTIVE_STATUSES)),
            ).fetchone()
            yield bool(locked and locked["locked"] and active is None)

    def claim(self) -> SandboxJobRecord | None:
        """Take the oldest queued job. SKIP LOCKED keeps slots from colliding.

        Returns:
            The claimed job, now running, or None when nothing claimable is queued.
        """

        now = datetime.now(UTC)
        with self.pool.connection() as connection:
            row = connection.execute(
                f"""
                UPDATE sandbox_jobs
                SET status = 'running', started_at = %s
                WHERE id = (
                    SELECT id FROM sandbox_jobs
                    WHERE status = 'queued' AND NOT cancel_requested AND created_at >= %s
                    ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED
                )
                RETURNING {_COLUMNS}
                """,
                (now, now - timedelta(seconds=CLAIM_DEADLINE_SECONDS)),
            ).fetchone()
        return _record(row) if row is not None else None

    def get(self, job_id: UUID) -> SandboxJobRecord | None:
        """Return one job.

        Args:
            job_id: The job's id.

        Returns:
            The job, or None when it does not exist.
        """
        with self.pool.connection() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM sandbox_jobs WHERE id = %s",
                (job_id,),
            ).fetchone()
        return _record(row) if row is not None else None

    def finish(
        self,
        job_id: UUID,
        *,
        status: JobStatus,
        output: str,
        exit_code: int | None = None,
        truncated: bool = False,
        artifact_names: list[str] | None = None,
    ) -> SandboxJobRecord | None:
        """Record a job's terminal status and result.

        A job that a stop asked to cancel is stored as ``cancelled`` with no
        files. A job that has already finished is left as it is.

        Args:
            job_id: The job's id.
            status: The terminal status.
            output: The command's output, or the reason it failed.
            exit_code: The command's exit code, if it has one.
            truncated: Whether the output was cut at the byte limit.
            artifact_names: The output files accepted for delivery.

        Returns:
            The job as stored afterwards, or None when it does not exist.

        Raises:
            ValueError: ``status`` is not a terminal status.
        """
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"Sandbox job status is not terminal: {status}")
        with self.pool.connection() as connection:
            row = connection.execute(
                f"""
                UPDATE sandbox_jobs
                SET status = CASE WHEN cancel_requested THEN 'cancelled' ELSE %s END,
                    output = %s, exit_code = %s, truncated = %s,
                    artifact_names = CASE WHEN cancel_requested THEN '[]'::jsonb ELSE %s END,
                    finished_at = %s
                WHERE id = %s AND status = ANY(%s)
                RETURNING {_COLUMNS}
                """,
                (
                    status,
                    output,
                    exit_code,
                    truncated,
                    Jsonb(artifact_names or []),
                    datetime.now(UTC),
                    job_id,
                    list(ACTIVE_STATUSES),
                ),
            ).fetchone()
        return _record(row) if row is not None else self.get(job_id)

    def cancel_run(self, run_id: str) -> int:
        """Cancel queued jobs; running slots retain ownership until cleanup.

        Running jobs are only marked, and their worker slot stops them.

        Args:
            run_id: The run whose jobs to cancel.

        Returns:
            How many of the run's jobs are still running.
        """
        with self.pool.connection() as connection, connection.transaction():
            connection.execute(
                "UPDATE sandbox_jobs SET cancel_requested = true, "
                "status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END, "
                "finished_at = CASE WHEN status = 'queued' THEN %s ELSE finished_at END "
                "WHERE run_id = %s AND status = ANY(%s)",
                (datetime.now(UTC), run_id, list(ACTIVE_STATUSES)),
            )
            row = connection.execute(
                "SELECT count(*) AS count FROM sandbox_jobs WHERE run_id = %s AND status = ANY(%s)",
                (run_id, list(ACTIVE_STATUSES)),
            ).fetchone()
            return row["count"]

    def fail_active(self, output: str) -> int:
        """Fail every row left by a previous worker process.

        Args:
            output: The reason stored on each row.

        Returns:
            How many rows were failed.
        """

        with self.pool.connection() as connection:
            return connection.execute(
                "UPDATE sandbox_jobs SET status = CASE WHEN cancel_requested THEN 'cancelled' ELSE 'failed' END, output = %s, "
                "finished_at = %s WHERE status = ANY(%s)",
                (output, datetime.now(UTC), list(ACTIVE_STATUSES)),
            ).rowcount

    def fail_stale_queued(self) -> int:
        """Fail queued jobs that no worker claimed within the claim deadline.

        Returns:
            How many jobs were failed.
        """
        now = datetime.now(UTC)
        with self.pool.connection() as connection:
            return connection.execute(
                "UPDATE sandbox_jobs SET status = 'failed', "
                "output = 'Sandbox worker was unavailable', finished_at = %s "
                "WHERE status = 'queued' AND created_at < %s",
                (now, now - timedelta(seconds=CLAIM_DEADLINE_SECONDS)),
            ).rowcount

    def fail_queued(self, job_id: UUID, output: str) -> SandboxJobRecord | None:
        """Fail one job if it is still queued.

        Args:
            job_id: The job's id.
            output: The reason stored on the row.

        Returns:
            The failed job, or None when it was not queued.
        """
        with self.pool.connection() as connection:
            row = connection.execute(
                f"""
                UPDATE sandbox_jobs SET status = 'failed', output = %s, finished_at = %s
                WHERE id = %s AND status = 'queued'
                RETURNING {_COLUMNS}
                """,
                (output, datetime.now(UTC), job_id),
            ).fetchone()
        return _record(row) if row is not None else None

    def active_workspaces(self) -> set[str]:
        """Return the workspaces that have a queued or running job.

        Returns:
            Their workspace ids.
        """
        with self.pool.connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT workspace_id FROM sandbox_jobs WHERE status = ANY(%s)",
                (list(ACTIVE_STATUSES),),
            ).fetchall()
        return {row["workspace_id"] for row in rows}

    def cleanup(self, *, older_than: datetime) -> int:
        """Delete finished jobs.

        Args:
            older_than: Delete jobs that finished before this time.

        Returns:
            How many rows were deleted.
        """
        with self.pool.connection() as connection:
            return connection.execute(
                "DELETE FROM sandbox_jobs WHERE status = ANY(%s) AND finished_at < %s",
                (list(TERMINAL_STATUSES), older_than),
            ).rowcount
