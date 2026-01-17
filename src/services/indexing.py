"""Indexing service - job management and pipeline orchestration."""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.core.config import settings
from src.db.models import IndexingJob, JobStage, JobStatus
from src.db.session import get_db_context

# Add kREPS-rag/src to path for imports
KREPS_RAG_SRC = Path(__file__).parent.parent.parent / "kREPS-rag" / "src"
if str(KREPS_RAG_SRC) not in sys.path:
    sys.path.insert(0, str(KREPS_RAG_SRC))

logger = logging.getLogger(__name__)


class IndexingService:
    """Manages indexing jobs and orchestrates the pipeline."""

    @staticmethod
    def calculate_progress(stage: JobStage, processed: int, total: int | None) -> float:
        """Calculate overall progress percentage based on stage."""
        start, end = settings.STAGE_WEIGHTS.get(stage.value, (0.0, 0.0))
        if total is None or total == 0:
            return start
        stage_progress = min(processed / total, 1.0)
        return start + (end - start) * stage_progress

    @staticmethod
    async def update_progress(
        job_id: str,
        stage: JobStage,
        processed: int,
        total: int | None,
        *,
        is_documents: bool = True,
    ) -> None:
        """Update job progress in database."""
        async with get_db_context() as db:
            result = await db.execute(
                select(IndexingJob).where(IndexingJob.id == job_id).with_for_update()
            )
            job = result.scalar_one_or_none()
            if job is None:
                logger.error(f"Job {job_id} not found")
                return

            job.current_stage = stage
            if is_documents:
                job.total_documents = total
                job.processed_documents = processed
            else:
                job.total_chunks = total
                job.processed_chunks = processed

            job.progress_percent = IndexingService.calculate_progress(
                stage, processed, total
            )
            await db.commit()

    @staticmethod
    async def mark_started(job_id: str) -> None:
        """Mark job as running."""
        async with get_db_context() as db:
            result = await db.execute(
                select(IndexingJob).where(IndexingJob.id == job_id).with_for_update()
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now(timezone.utc)
                await db.commit()

    @staticmethod
    async def mark_completed(job_id: str) -> None:
        """Mark job as completed."""
        async with get_db_context() as db:
            result = await db.execute(
                select(IndexingJob).where(IndexingJob.id == job_id).with_for_update()
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = JobStatus.COMPLETED
                job.current_stage = JobStage.FINALIZE
                job.progress_percent = 100.0
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()

    @staticmethod
    async def mark_failed(job_id: str, error: str) -> None:
        """Mark job as failed."""
        async with get_db_context() as db:
            result = await db.execute(
                select(IndexingJob).where(IndexingJob.id == job_id).with_for_update()
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = JobStatus.FAILED
                job.error_message = error
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()

    @classmethod
    async def run_pipeline(cls, job_id: str, source_path: str, index_name: str) -> None:
        """Execute the full indexing pipeline."""
        try:
            await cls.mark_started(job_id)

            # Stage 1: Ingest
            await cls.update_progress(job_id, JobStage.INGEST, 0, None, is_documents=True)
            documents = await cls._run_ingest(source_path, job_id)

            # Stage 2: Chunk
            await cls.update_progress(job_id, JobStage.CHUNK, 0, None, is_documents=False)
            chunks = await cls._run_chunk(documents, job_id)

            # Stage 3: Embed
            await cls.update_progress(job_id, JobStage.EMBED, 0, None, is_documents=False)
            embedded_chunks = await cls._run_embed(chunks, job_id)

            # Stage 4: Index
            await cls.update_progress(job_id, JobStage.INDEX, 0, None, is_documents=False)
            await cls._run_index(embedded_chunks, index_name, job_id)

            # Stage 5: Finalize
            await cls.update_progress(job_id, JobStage.FINALIZE, 0, 1, is_documents=False)
            await cls._run_finalize(index_name, job_id)
            await cls.update_progress(job_id, JobStage.FINALIZE, 1, 1, is_documents=False)

            await cls.mark_completed(job_id)
            logger.info(f"Job {job_id} completed")

        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            await cls.mark_failed(job_id, str(e))

    # --- Pipeline Stage Implementations ---

    @classmethod
    async def _run_ingest(cls, source_path: str, job_id: str) -> list[dict[str, Any]]:
        """Ingest documents from source path using kREPS-rag DocumentLoader."""
        from ingest import DocumentLoader

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path not found: {source_path}")

        # Use DocumentLoader with custom source path
        loader = DocumentLoader(docs_dir=Path(source_path))

        # Run in thread pool to avoid blocking async loop
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, loader.load_all)

        total = len(documents)
        await cls.update_progress(job_id, JobStage.INGEST, total, total, is_documents=True)

        logger.info(f"Ingested {total} documents from {source_path}")
        return documents

    @classmethod
    async def _run_chunk(
        cls, documents: list[dict[str, Any]], job_id: str
    ) -> list[dict[str, Any]]:
        """Chunk documents using kREPS-rag TextChunker."""
        from chunking import TextChunker

        chunker = TextChunker()

        # Run in thread pool to avoid blocking async loop
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, chunker.chunk_documents, documents)

        total = len(chunks)
        await cls.update_progress(job_id, JobStage.CHUNK, total, total, is_documents=False)

        logger.info(f"Created {total} chunks from {len(documents)} documents")
        return chunks

    @classmethod
    async def _run_embed(
        cls, chunks: list[dict[str, Any]], job_id: str
    ) -> list[dict[str, Any]]:
        """Generate embeddings using kREPS-rag EmbeddingEngine (Ollama)."""
        from embed import get_embedding_engine

        engine = get_embedding_engine()

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        total = len(texts)

        # Run embedding in thread pool (blocking HTTP calls)
        loop = asyncio.get_event_loop()

        # Process in batches with progress updates
        batch_size = engine.batch_size
        all_embeddings: list[list[float]] = []

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await loop.run_in_executor(
                None, engine.embed_texts, batch
            )
            all_embeddings.extend(batch_embeddings)

            # Update progress
            processed = min(i + batch_size, total)
            await cls.update_progress(
                job_id, JobStage.EMBED, processed, total, is_documents=False
            )

        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, all_embeddings):
            embedded_chunk = chunk.copy()
            embedded_chunk["embedding"] = embedding
            embedded_chunks.append(embedded_chunk)

        logger.info(f"Generated embeddings for {total} chunks")
        return embedded_chunks

    @classmethod
    async def _run_index(
        cls, chunks: list[dict[str, Any]], index_name: str, job_id: str
    ) -> None:
        """Build FAISS and BM25 indices using kREPS-rag indexing modules."""
        from index_faiss import FAISSIndex
        from index_bm25 import BM25Index
        from config import STORAGE_DIR

        # Create index-specific storage directory
        index_dir = STORAGE_DIR / "indices" / index_name
        faiss_dir = index_dir / "faiss"
        bm25_dir = index_dir / "bm25"
        faiss_dir.mkdir(parents=True, exist_ok=True)
        bm25_dir.mkdir(parents=True, exist_ok=True)

        total = len(chunks)
        loop = asyncio.get_event_loop()

        # Build FAISS index
        logger.info(f"Building FAISS index for {total} chunks")
        faiss_index = FAISSIndex()
        faiss_index.index_path = faiss_dir / "index.faiss"
        faiss_index.chunks_path = faiss_dir / "chunks.pkl"

        await loop.run_in_executor(None, faiss_index.build_index, chunks)
        await cls.update_progress(job_id, JobStage.INDEX, total // 2, total, is_documents=False)

        # Build BM25 index
        logger.info(f"Building BM25 index for {total} chunks")
        bm25_index = BM25Index()
        bm25_index.index_path = bm25_dir / "bm25.pkl"
        bm25_index.chunks_path = bm25_dir / "chunks.pkl"

        await loop.run_in_executor(None, bm25_index.build_index, chunks)
        await cls.update_progress(job_id, JobStage.INDEX, total, total, is_documents=False)

        logger.info(f"Built FAISS and BM25 indices for index '{index_name}'")

    @classmethod
    async def _run_finalize(cls, index_name: str, job_id: str) -> None:
        """Finalize indexing - indices already saved by build methods."""
        from config import STORAGE_DIR

        index_dir = STORAGE_DIR / "indices" / index_name

        # Verify indices exist
        faiss_index_path = index_dir / "faiss" / "index.faiss"
        bm25_index_path = index_dir / "bm25" / "bm25.pkl"

        if not faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")
        if not bm25_index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {bm25_index_path}")

        logger.info(f"Finalized index '{index_name}' at {index_dir}")
