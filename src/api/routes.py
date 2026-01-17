"""API route handlers."""

import logging
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.models import IndexingJob, JobStatus
from src.db.session import get_db
from src.schemas.api import (
    ErrorResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
)
from src.services.indexing import IndexingService
from src.services.query import QueryService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """Check API and database health."""
    db_status = "healthy"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "unhealthy"

    return HealthResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        database=db_status,
        version=settings.API_VERSION,
    )


@router.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    tags=["Indexing"],
)
async def create_index_job(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> IndexResponse:
    """Create a new indexing job."""
    # Check for existing active job
    result = await db.execute(
        select(IndexingJob).where(
            IndexingJob.index_name == request.index_name,
            IndexingJob.status.in_([JobStatus.PENDING, JobStatus.RUNNING]),
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Active job exists for index '{request.index_name}'",
        )

    # Validate source path
    if not os.path.exists(request.source_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Source path not found: {request.source_path}",
        )

    # Create job
    job = IndexingJob(
        id=str(uuid4()),
        source_path=request.source_path,
        index_name=request.index_name,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Schedule background task
    background_tasks.add_task(
        IndexingService.run_pipeline,
        job.id,
        request.source_path,
        request.index_name,
    )

    logger.info(f"Created job {job.id} for index '{request.index_name}'")

    return IndexResponse(job_id=job.id, message="Indexing job created")


@router.get(
    "/index/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Indexing"],
)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    """Get indexing job status."""
    result = await db.execute(select(IndexingJob).where(IndexingJob.id == job_id))
    job = result.scalar_one_or_none()

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        current_stage=job.current_stage,
        progress_percent=job.progress_percent,
        source_path=job.source_path,
        index_name=job.index_name,
        total_documents=job.total_documents,
        processed_documents=job.processed_documents,
        total_chunks=job.total_chunks,
        processed_chunks=job.processed_chunks,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.delete(
    "/index/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    tags=["Indexing"],
)
async def cancel_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Cancel an indexing job."""
    result = await db.execute(
        select(IndexingJob).where(IndexingJob.id == job_id).with_for_update()
    )
    job = result.scalar_one_or_none()

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel job with status: {job.status}",
        )

    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.now(timezone.utc)
    await db.commit()

    logger.info(f"Cancelled job {job_id}")


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["Query"],
)
async def query_index(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """Query the RAG system."""
    return await QueryService.process(request, db)


@router.get("/indices", tags=["Indexing"])
async def list_indices(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """List completed indices."""
    result = await db.execute(
        select(IndexingJob).where(IndexingJob.status == JobStatus.COMPLETED)
    )
    jobs = result.scalars().all()

    return {
        "indices": [
            {
                "name": job.index_name,
                "source_path": job.source_path,
                "total_documents": job.total_documents,
                "total_chunks": job.total_chunks,
                "created_at": job.completed_at.isoformat() if job.completed_at else None,
            }
            for job in jobs
        ]
    }
