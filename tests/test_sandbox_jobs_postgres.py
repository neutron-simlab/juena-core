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
