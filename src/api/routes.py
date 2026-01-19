"""API route handlers with robust error handling."""

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select, text, desc
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.models import IndexingJob, JobStage, JobStatus, QueryAuditLog
from src.db.session import get_db
from src.schemas.api import (
    AuditLogEntry,
    AuditLogsResponse,
    DocumentInfo,
    ErrorResponse,
    FileUploadResponse,
    HealthResponse,
    IndexRequest,
    IndexResetResponse,
    IndexResponse,
    IndexVerifyResponse,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    SystemStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ============================================
# Storage paths - CENTRALIZED
# ============================================
KREPS_RAG_ROOT = Path(__file__).parent.parent.parent / "kREPS-rag"
STORAGE_DIR = KREPS_RAG_ROOT / "storage"
DATA_DIR = KREPS_RAG_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
PROCESSED_DIR = DATA_DIR / "processed"
FAISS_DIR = STORAGE_DIR / "faiss"
BM25_DIR = STORAGE_DIR / "bm25"
INDICES_DIR = STORAGE_DIR / "indices"  # Where IndexingService writes
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"


def ensure_directories():
    """Create required directories if they don't exist."""
    for d in [RAW_DOCS_DIR, PROCESSED_DIR, FAISS_DIR, BM25_DIR, INDICES_DIR]:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directory {d}: {e}")


def safe_count_lines(file_path: Path) -> int:
    """Safely count lines in a file. Returns 0 on any error."""
    if not file_path.exists():
        return 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def safe_count_files(dir_path: Path) -> int:
    """Safely count files in a directory. Returns 0 on any error."""
    if not dir_path.exists():
        return 0
    try:
        return len([f for f in dir_path.iterdir() if f.is_file() and not f.name.startswith('.')])
    except Exception:
        return 0


def check_index_exists() -> bool:
    """Check if any index exists (FAISS or in indices dir)."""
    try:
        # Check direct FAISS dir
        if (FAISS_DIR / "index.faiss").exists():
            return True
        if (FAISS_DIR / "chunks.pkl").exists():
            return True

        # Check indices subdirectories
        if INDICES_DIR.exists():
            for subdir in INDICES_DIR.iterdir():
                if subdir.is_dir():
                    faiss_path = subdir / "faiss" / "index.faiss"
                    if faiss_path.exists():
                        return True
        return False
    except Exception:
        return False


# ============================================
# HEALTH ENDPOINT - MUST NEVER CRASH
# ============================================

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API health. NEVER crashes - always returns 200.
    Does not require database connection.
    """
    db_status = "unknown"
    db_error = None

    # Try to check database without failing
    try:
        from src.db.session import async_session_factory
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
            db_status = "healthy"
    except Exception as e:
        db_status = "unhealthy"
        db_error = str(e)
        logger.warning(f"Database health check failed: {e}")

    return HealthResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        database=db_status,
        version=settings.API_VERSION,
    )


# ============================================
# STATUS ENDPOINT - NEVER CRASHES
# ============================================

@router.get("/status", response_model=SystemStatusResponse, tags=["Status"])
async def get_system_status():
    """
    Get comprehensive system status with REAL metrics.
    Never crashes - returns zeros on errors.
    """
    ensure_directories()

    # Check DB connection (without crashing)
    db_connected = False
    try:
        from src.db.session import async_session_factory
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
            db_connected = True
    except Exception as e:
        logger.warning(f"Database connection check failed: {e}")

    # Check Ollama connection
    ollama_connected = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            ollama_connected = resp.status_code == 200
    except Exception:
        pass

    # Count real chunks from chunks.jsonl
    total_chunks = safe_count_lines(CHUNKS_FILE)

    # Count documents in raw_docs
    total_documents = safe_count_files(RAW_DOCS_DIR)

    # Check if index is ready
    index_ready = check_index_exists()

    # Get last indexing job (safely)
    last_job = None
    if db_connected:
        try:
            from src.db.session import async_session_factory
            async with async_session_factory() as session:
                result = await session.execute(
                    select(IndexingJob).order_by(desc(IndexingJob.created_at)).limit(1)
                )
                job = result.scalar_one_or_none()
                if job:
                    last_job = {
                        "id": job.id,
                        "status": job.status.value,
                        "stage": job.current_stage.value,
                        "progress": job.progress_percent,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                    }
        except Exception as e:
            logger.warning(f"Could not get last job: {e}")

    # Determine system status
    if not db_connected:
        system = "error"
    elif not index_ready and total_chunks == 0:
        system = "ready"  # Ready but no index yet
    else:
        system = "ready"

    return SystemStatusResponse(
        system=system,
        ollama_connected=ollama_connected,
        index_ready=index_ready,
        db_connected=db_connected,
        total_documents=total_documents,
        total_chunks=total_chunks,
        storage_paths={
            "raw_docs": str(RAW_DOCS_DIR),
            "processed": str(PROCESSED_DIR),
            "faiss": str(FAISS_DIR),
            "bm25": str(BM25_DIR),
            "indices": str(INDICES_DIR),
        },
        last_index_job=last_job,
    )


# ============================================
# INDEX RESET - ACTUALLY DELETES EVERYTHING
# ============================================

@router.post("/index/reset", response_model=IndexResetResponse, tags=["Indexing"])
async def reset_index(db: AsyncSession = Depends(get_db)) -> IndexResetResponse:
    """
    Clear ALL persisted indices, chunks, and DB records.
    This is a REAL reset - deletes actual files.
    """
    ensure_directories()

    files_deleted = 0
    chunks_deleted = 0
    dirs_cleared = []
    errors = []

    # 1. Clear FAISS directory
    try:
        if FAISS_DIR.exists():
            for f in FAISS_DIR.iterdir():
                if f.name.startswith('.'):  # Skip hidden files like .gitkeep
                    continue
                try:
                    if f.is_file():
                        f.unlink()
                        files_deleted += 1
                    elif f.is_dir():
                        shutil.rmtree(f)
                        files_deleted += 1
                except Exception as e:
                    errors.append(f"faiss/{f.name}: {e}")
            dirs_cleared.append("faiss")
    except Exception as e:
        errors.append(f"faiss: {e}")

    # 2. Clear BM25 directory
    try:
        if BM25_DIR.exists():
            for f in BM25_DIR.iterdir():
                if f.name.startswith('.'):
                    continue
                try:
                    if f.is_file():
                        f.unlink()
                        files_deleted += 1
                    elif f.is_dir():
                        shutil.rmtree(f)
                        files_deleted += 1
                except Exception as e:
                    errors.append(f"bm25/{f.name}: {e}")
            dirs_cleared.append("bm25")
    except Exception as e:
        errors.append(f"bm25: {e}")

    # 3. Clear indices directory (where IndexingService writes)
    try:
        if INDICES_DIR.exists():
            for f in INDICES_DIR.iterdir():
                if f.name.startswith('.'):
                    continue
                try:
                    if f.is_file():
                        f.unlink()
                        files_deleted += 1
                    elif f.is_dir():
                        shutil.rmtree(f)
                        files_deleted += 1
                except Exception as e:
                    errors.append(f"indices/{f.name}: {e}")
            dirs_cleared.append("indices")
    except Exception as e:
        errors.append(f"indices: {e}")

    # 4. Clear chunks.jsonl
    try:
        if CHUNKS_FILE.exists():
            chunks_deleted = safe_count_lines(CHUNKS_FILE)
            CHUNKS_FILE.unlink()
            files_deleted += 1
    except Exception as e:
        errors.append(f"chunks.jsonl: {e}")

    # 5. Clear other processed files
    try:
        if PROCESSED_DIR.exists():
            for f in PROCESSED_DIR.iterdir():
                if f.name.startswith('.'):
                    continue
                try:
                    if f.is_file():
                        f.unlink()
                        files_deleted += 1
                except Exception as e:
                    errors.append(f"processed/{f.name}: {e}")
            dirs_cleared.append("processed")
    except Exception as e:
        errors.append(f"processed: {e}")

    # 6. Clear indexing jobs from DB
    tables_cleared = []
    try:
        await db.execute(text("DELETE FROM indexing_jobs"))
        tables_cleared.append("indexing_jobs")
    except Exception as e:
        logger.warning(f"Could not clear indexing_jobs: {e}")
        errors.append(f"db:indexing_jobs: {e}")

    # 7. Optionally clear audit logs
    try:
        await db.execute(text("DELETE FROM query_audit_logs"))
        tables_cleared.append("query_audit_logs")
    except Exception as e:
        logger.warning(f"Could not clear query_audit_logs: {e}")

    try:
        await db.commit()
    except Exception as e:
        errors.append(f"db commit: {e}")

    logger.info(f"Index reset: {files_deleted} files deleted, {chunks_deleted} chunks cleared")

    return IndexResetResponse(
        ok=len(errors) == 0,
        cleared={
            "files_deleted": files_deleted,
            "chunks_deleted": chunks_deleted,
            "directories_cleared": dirs_cleared,
            "db_tables_cleared": tables_cleared,
            "errors": errors if errors else None,
        }
    )


# ============================================
# INDEX VERIFY
# ============================================

@router.get("/index/verify", response_model=IndexVerifyResponse, tags=["Indexing"])
async def verify_index(db: AsyncSession = Depends(get_db)) -> IndexVerifyResponse:
    """
    Verify indexed documents with metadata.
    Returns empty list if no index exists (NOT an error).
    """
    ensure_directories()

    documents: list[DocumentInfo] = []
    total_chunks = 0

    # Check if index is ready
    index_ready = check_index_exists()

    # Parse chunks.jsonl to extract document info
    if CHUNKS_FILE.exists():
        doc_chunks: dict[str, list[dict]] = {}

        try:
            with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        total_chunks += 1

                        # Group by document
                        doc_name = chunk.get("metadata", {}).get("document", "unknown")
                        if doc_name not in doc_chunks:
                            doc_chunks[doc_name] = []
                        doc_chunks[doc_name].append(chunk)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading chunks file: {e}")

        # Build document info from chunks
        for doc_name, chunks in doc_chunks.items():
            if not chunks:
                continue

            # Get metadata from first chunk
            first_chunk = chunks[0]
            metadata = first_chunk.get("metadata", {})

            # Compute file hash from content
            content_hash = hashlib.sha256(
                "".join(c.get("text", "") for c in chunks).encode()
            ).hexdigest()[:16]

            documents.append(DocumentInfo(
                filename=doc_name,
                file_hash=content_hash,
                security_level=metadata.get("security_level", "public"),
                document_type=metadata.get("document_type", "other"),
                language=metadata.get("language", "en"),
                chunks_count=len(chunks),
                indexed_at=None,
            ))

    return IndexVerifyResponse(
        index_ready=index_ready,
        documents=documents,
        total_documents=len(documents),
        total_chunks=total_chunks,
    )


# ============================================
# FILE UPLOAD
# ============================================

@router.post("/index/upload", response_model=FileUploadResponse, tags=["Indexing"])
async def upload_files(files: list[UploadFile] = File(...)) -> FileUploadResponse:
    """Upload files to raw_docs directory for indexing."""
    ensure_directories()

    uploaded_files = []
    errors = []

    for file in files:
        if not file.filename:
            errors.append("File without name skipped")
            continue

        # Validate file extension
        allowed_extensions = {'.pdf', '.txt', '.md', '.docx', '.doc'}
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            errors.append(f"{file.filename}: unsupported format ({ext})")
            continue

        try:
            file_path = RAW_DOCS_DIR / file.filename

            # Read and save file
            content = await file.read()
            with open(file_path, 'wb') as f:
                f.write(content)

            uploaded_files.append(file.filename)
            logger.info(f"Uploaded file: {file.filename}")
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    return FileUploadResponse(
        uploaded=len(uploaded_files),
        files=uploaded_files,
        errors=errors,
    )


# ============================================
# INDEXING ENDPOINTS
# ============================================

@router.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    tags=["Indexing"],
)
async def create_index_job(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> IndexResponse:
    """Create a new indexing job."""
    from src.services.indexing import IndexingService

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


@router.post("/index/start", tags=["Indexing"])
async def start_indexing_simple(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    force: bool = False,
) -> dict:
    """
    Start indexing with default paths (for UI simplicity).
    Uses kREPS-rag/data/raw_docs as source.
    """
    from src.services.indexing import IndexingService

    ensure_directories()

    source_path = str(RAW_DOCS_DIR)
    index_name = "default"

    # Check for existing active job (unless force)
    if not force:
        result = await db.execute(
            select(IndexingJob).where(
                IndexingJob.index_name == index_name,
                IndexingJob.status.in_([JobStatus.PENDING, JobStatus.RUNNING]),
            )
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Active indexing job already exists",
            )

    # Validate source path has files
    doc_count = safe_count_files(RAW_DOCS_DIR)
    if doc_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents found in raw_docs directory. Upload files first.",
        )

    # Create job
    job = IndexingJob(
        id=str(uuid4()),
        source_path=source_path,
        index_name=index_name,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Schedule background task
    background_tasks.add_task(
        IndexingService.run_pipeline,
        job.id,
        source_path,
        index_name,
    )

    logger.info(f"Started indexing job {job.id}")

    return {"job_id": job.id, "status": "started", "message": "Indexing started"}


@router.get("/index/status", tags=["Indexing"])
async def get_index_status(db: AsyncSession = Depends(get_db)) -> dict:
    """Get current indexing status (for UI polling)."""
    ensure_directories()

    # Get most recent job
    try:
        result = await db.execute(
            select(IndexingJob).order_by(desc(IndexingJob.created_at)).limit(1)
        )
        job = result.scalar_one_or_none()
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        job = None

    if not job:
        # No jobs yet - return idle state with real counts
        total_chunks = safe_count_lines(CHUNKS_FILE)
        total_docs = safe_count_files(RAW_DOCS_DIR)

        return {
            "stage": "idle",
            "progress": 100 if total_chunks > 0 else 0,
            "documents_total": total_docs,
            "documents_processed": total_docs if total_chunks > 0 else 0,
            "chunks_created": total_chunks,
            "elapsed_ms": 0,
            "errors": [],
        }

    # Map job stage to UI stage
    stage_map = {
        JobStage.QUEUED: "idle",
        JobStage.INGEST: "ingest",
        JobStage.CHUNK: "chunk",
        JobStage.EMBED: "embed",
        JobStage.INDEX: "index",
        JobStage.FINALIZE: "complete",
    }

    stage = stage_map.get(job.current_stage, "idle")
    if job.status == JobStatus.COMPLETED:
        stage = "complete"
    elif job.status == JobStatus.FAILED:
        stage = "error"

    # Calculate elapsed time
    elapsed_ms = 0
    if job.started_at:
        end_time = job.completed_at or datetime.now(timezone.utc)
        elapsed_ms = int((end_time - job.started_at).total_seconds() * 1000)

    return {
        "stage": stage,
        "progress": job.progress_percent,
        "documents_total": job.total_documents or 0,
        "documents_processed": job.processed_documents,
        "chunks_created": job.total_chunks or 0,
        "current_file": None,
        "elapsed_ms": elapsed_ms,
        "estimated_remaining_ms": 0,
        "errors": [job.error_message] if job.error_message else [],
    }


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


@router.get("/indices", tags=["Indexing"])
async def list_indices(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """List completed indices."""
    try:
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
    except Exception as e:
        logger.error(f"Error listing indices: {e}")
        return {"indices": [], "error": str(e)}


# ============================================
# QUERY ENDPOINT
# ============================================

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
    """Query the RAG system with deterministic behavior."""
    from src.services.query import QueryService
    return await QueryService.process(request, db)


# ============================================
# AUDIT ENDPOINTS
# ============================================

@router.get("/audit/logs", response_model=AuditLogsResponse, tags=["Audit"])
async def get_audit_logs(
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
) -> AuditLogsResponse:
    """Get recent audit logs."""
    try:
        # Get total count
        count_result = await db.execute(select(func.count(QueryAuditLog.id)))
        total_count = count_result.scalar() or 0

        # Get logs
        result = await db.execute(
            select(QueryAuditLog)
            .order_by(desc(QueryAuditLog.created_at))
            .limit(limit)
            .offset(offset)
        )
        logs = result.scalars().all()

        return AuditLogsResponse(
            logs=[
                AuditLogEntry(
                    id=log.id,
                    timestamp=log.created_at,
                    query_hash=log.query_hash,
                    clearance_level=log.user_clearance_level,
                    chunks_retrieved=log.retrieved_count,
                    chunks_blocked=log.retrieved_count - log.used_count,
                    top_score=log.top_score,
                    refused=log.refused,
                    refusal_reason=log.refusal_reason.value if log.refusal_reason else None,
                    latency_ms=log.latency_ms,
                )
                for log in logs
            ],
            total_count=total_count,
        )
    except Exception as e:
        logger.error(f"Error getting audit logs: {e}")
        return AuditLogsResponse(logs=[], total_count=0)
