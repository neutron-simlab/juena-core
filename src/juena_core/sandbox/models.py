"""Optional SQLAlchemy model for the Postgres sandbox job queue."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Index, Integer, JSON, String, Text, Uuid, text
from sqlalchemy.orm import Mapped, mapped_column

from juena_core.sandbox.constants import ACTIVE_STATUSES
from juena_core.server.database.models import Base, utc_now

__all__ = ["SandboxJob"]

_ACTIVE_JOB_PREDICATE = text(
    "status IN ({})".format(", ".join(f"'{status}'" for status in ACTIVE_STATUSES))
)


class SandboxJob(Base):
    """Durable metadata shared by the API and the rootless-Podman worker."""

    __tablename__ = "sandbox_jobs"
    __table_args__ = (
        Index("ix_sandbox_jobs_status_created", "status", "created_at"),
        Index(
            "uq_sandbox_jobs_active_tenant",
            "tenant_id",
            unique=True,
            postgresql_where=_ACTIVE_JOB_PREDICATE,
        ),
        Index(
            "uq_sandbox_jobs_active_workspace",
            "workspace_id",
            unique=True,
            postgresql_where=_ACTIVE_JOB_PREDICATE,
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    command: Mapped[str] = mapped_column(Text, nullable=False)
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    max_output_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    output: Mapped[str] = mapped_column(Text, nullable=False, default="")
    exit_code: Mapped[int | None] = mapped_column(Integer)
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    artifact_names: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
