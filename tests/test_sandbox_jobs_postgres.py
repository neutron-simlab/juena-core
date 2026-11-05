"""Durable sandbox queue contracts against the throwaway test Postgres."""

from __future__ import annotations

import subprocess
import sys


def test_queue_admission_claim_result_and_worker_lock_round_trip() -> None:
    """Exercise the real SQL without importing the optional model in pytest.

    The subprocess is intentional: importing ``sandbox.models`` attaches its table to
    process-global core metadata, while the main suite also proves the base package owns
    exactly three tables until an application opts in.
    """

    code = r'''
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, text

from juena_core.sandbox.jobs import SandboxJobs, hold_worker_lock
from juena_core.sandbox.models import SandboxJob

dsn = "postgresql+psycopg://juena_core_test:juena_core_test@127.0.0.1:55432/juena_core_test"
engine = create_engine(dsn)
SandboxJob.__table__.create(engine, checkfirst=True)
with engine.begin() as connection:
    connection.execute(text("TRUNCATE TABLE sandbox_jobs"))

jobs = SandboxJobs(dsn, concurrency=1)
jobs.open()
try:
    first = jobs.create(
        tenant_id="1" * 64,
        workspace_id="a" * 64,
        command="printf first",
        timeout_seconds=10,
        max_output_bytes=1000,
    )
    assert first is not None and first.status == "queued"

    # Capacity is checked transactionally before a second active row is inserted.
    assert jobs.create(
        tenant_id="2" * 64,
        workspace_id="b" * 64,
        command="printf blocked",
        timeout_seconds=10,
        max_output_bytes=1000,
    ) is None

    claimed = jobs.claim()
    assert claimed is not None and claimed.id == first.id
    assert claimed.status == "running"
    finished = jobs.finish(
        first.id,
        status="completed",
        output="first",
        exit_code=0,
        artifact_names=["result.txt"],
    )
    assert finished is not None
    assert finished.output == "first"
    assert finished.artifact_names == ("result.txt",)

    second = jobs.create(
        tenant_id="2" * 64,
        workspace_id="b" * 64,
        command="printf second",
        timeout_seconds=10,
        max_output_bytes=1000,
    )
    assert second is not None
    jobs.finish(second.id, status="failed", output="test cleanup")

    assert jobs.worker_is_alive() is False
    with hold_worker_lock(dsn) as acquired:
        assert acquired is True
        assert jobs.worker_is_alive() is True

    removed = jobs.cleanup(older_than=datetime.now(UTC) + timedelta(seconds=1))
    assert removed == 2

    # Queued cancellation wins against claiming, without occupying capacity.
    queued = jobs.create(tenant_id="1"*64, workspace_id="a"*64,
        command="must not start", timeout_seconds=10, max_output_bytes=1000,
        run_id="queued-run")
    assert jobs.cancel_run("queued-run") == 0
    assert jobs.get(queued.id).status == "cancelled"
    assert jobs.claim() is None

    # Running cancellation retains the workspace until the worker acknowledges
    # container cleanup. A late successful result cannot publish artifacts.
    running = jobs.create(tenant_id="1"*64, workspace_id="a"*64,
        command="running", timeout_seconds=10, max_output_bytes=1000,
        run_id="running-run")
    assert jobs.claim().id == running.id
    assert jobs.cancel_run("running-run") == 1
    assert jobs.get(running.id).status == "running"
    with jobs.workspace_guard("a"*64) as available:
        assert available is False
    finished = jobs.finish(running.id, status="completed", output="late result", artifact_names=["unverified.png"])
    assert finished.status == "cancelled"
    assert finished.artifact_names == ()
    assert jobs.cancel_run("running-run") == 0
    with jobs.workspace_guard("a"*64) as available:
        assert available is True

    # Worker recovery preserves cancellation rather than making it runnable.
    recovery = jobs.create(tenant_id="1"*64, workspace_id="a"*64,
        command="interrupted", timeout_seconds=10, max_output_bytes=1000,
        run_id="recovery-run")
    assert jobs.claim().id == recovery.id
    assert jobs.cancel_run("recovery-run") == 1
    jobs.fail_active("worker restarted after cleanup")
    assert jobs.get(recovery.id).status == "cancelled"
    assert jobs.claim() is None
finally:
    jobs.close()
    engine.dispose()
'''
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
