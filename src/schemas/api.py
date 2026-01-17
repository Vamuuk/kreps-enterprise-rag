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
    index_name: str = Field(..., description="Target index", min_length=1, max_length=255)
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


class QueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    sources: list[RetrievedDocument] = Field(default_factory=list)
    query: str
    refused: bool = False
    refusal_reason: Optional[RefusalReason] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    database: str
    version: str
