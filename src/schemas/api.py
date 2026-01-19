"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.db.models import JobStage, JobStatus, RefusalReason, SecurityLevel


class IndexRequest(BaseModel):
    """Request to create an indexing job."""

    source_path: str = Field(
        ..., description="Path to documents", min_length=1, max_length=1024
    )
    index_name: str = Field(
        ...,
        description="Index identifier",
        min_length=1,
        max_length=255,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )


class IndexResponse(BaseModel):
    """Response after creating an indexing job."""

    job_id: str
    message: str


class JobStatusResponse(BaseModel):
    """Indexing job status and progress."""

    job_id: str
    status: JobStatus
    current_stage: JobStage
    progress_percent: float = Field(..., ge=0.0, le=100.0)
    source_path: str
    index_name: str
    total_documents: Optional[int] = None
    processed_documents: int
    total_chunks: Optional[int] = None
    processed_chunks: int
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class QueryRequest(BaseModel):
    """RAG query request."""

    query: str = Field(..., description="User query", min_length=1, max_length=4096)
    index_name: str = Field(default="default", description="Target index", min_length=1, max_length=255)
    top_k: int = Field(default=5, description="Number of results", ge=1, le=100)
    clearance_level: int = Field(
        default=SecurityLevel.PUBLIC.value,
        description="User clearance (0-4)",
        ge=0,
        le=4,
    )


class RetrievedDocument(BaseModel):
    """Retrieved document chunk."""

    content: str
    source: str
    score: float
    metadata: dict = Field(default_factory=dict)


class QueryMetrics(BaseModel):
    """Query timing breakdown."""

    retrieval_ms: int = 0
    ranking_ms: int = 0
    generation_ms: int = 0
    total_ms: int = 0
    chunks_retrieved: int = 0
    chunks_blocked: int = 0


class QueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    sources: list[RetrievedDocument] = Field(default_factory=list)
    query: str
    refused: bool = False
    refusal_reason: Optional[RefusalReason] = None
    confidence: str = "Low"
    metrics: Optional[QueryMetrics] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    database: str
    version: str


# ============================================
# NEW SCHEMAS FOR REAL DATA BINDING
# ============================================

class SystemStatusResponse(BaseModel):
    """Complete system status with real metrics."""

    system: str  # 'ready' | 'indexing' | 'error'
    ollama_connected: bool
    index_ready: bool
    db_connected: bool
    total_documents: int
    total_chunks: int
    storage_paths: dict[str, str]
    last_index_job: Optional[dict] = None


class IndexResetResponse(BaseModel):
    """Response after clearing indices."""

    ok: bool
    cleared: dict


class DocumentInfo(BaseModel):
    """Indexed document information."""

    filename: str
    file_hash: str
    security_level: str
    document_type: str
    language: str
    chunks_count: int
    indexed_at: Optional[datetime] = None


class IndexVerifyResponse(BaseModel):
    """Verification of indexed documents."""

    index_ready: bool
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class AuditLogEntry(BaseModel):
    """Single audit log entry."""

    id: str
    timestamp: datetime
    query_hash: str
    clearance_level: int
    chunks_retrieved: int
    chunks_blocked: int
    top_score: Optional[float]
    refused: bool
    refusal_reason: Optional[str]
    latency_ms: int

    model_config = ConfigDict(from_attributes=True)


class AuditLogsResponse(BaseModel):
    """List of audit log entries."""

    logs: list[AuditLogEntry]
    total_count: int


class FileUploadResponse(BaseModel):
    """Response after file upload."""

    uploaded: int
    files: list[str]
    errors: list[str]
