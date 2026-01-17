"""SQLAlchemy ORM models and enumerations."""

import enum
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Enum, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# --- Enumerations ---


class JobStatus(str, enum.Enum):
    """Job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStage(str, enum.Enum):
    """Indexing pipeline stages."""

    QUEUED = "queued"
    INGEST = "ingest"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"
    FINALIZE = "finalize"


class SecurityLevel(int, enum.Enum):
    """Document security clearance levels (higher = more restricted)."""

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    SECRET = 3
    TOP_SECRET = 4


class RefusalReason(str, enum.Enum):
    """Deterministic refusal reasons for audit trail."""

    NONE = "none"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_ALLOWED_CHUNKS = "no_allowed_chunks"
    BELOW_RELEVANCE_THRESHOLD = "below_relevance_threshold"
    INDEX_NOT_FOUND = "index_not_found"


# --- ORM Models ---


class IndexingJob(Base):
    """Tracks indexing job state and progress."""

    __tablename__ = "indexing_jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), nullable=False, default=JobStatus.PENDING, index=True
    )
    current_stage: Mapped[JobStage] = mapped_column(
        Enum(JobStage), nullable=False, default=JobStage.QUEUED
    )
    progress_percent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    source_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    index_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    total_documents: Mapped[Optional[int]] = mapped_column(nullable=True)
    processed_documents: Mapped[int] = mapped_column(nullable=False, default=0)
    total_chunks: Mapped[Optional[int]] = mapped_column(nullable=True)
    processed_chunks: Mapped[int] = mapped_column(nullable=False, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<IndexingJob(id={self.id}, status={self.status}, stage={self.current_stage})>"


class QueryAuditLog(Base):
    """Audit log for RAG queries. Query text is hashed for privacy."""

    __tablename__ = "query_audit_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    index_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_clearance_level: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieved_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    used_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    top_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    refused: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    refusal_reason: Mapped[RefusalReason] = mapped_column(
        Enum(RefusalReason), nullable=False, default=RefusalReason.NONE
    )
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    def __repr__(self) -> str:
        return f"<QueryAuditLog(id={self.id}, refused={self.refused})>"
