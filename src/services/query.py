"""Query service - RAG retrieval and response generation."""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.refusal import evaluate_evidence, get_refusal_message
from src.core.security import filter_chunks_by_clearance, hash_query
from src.db.models import QueryAuditLog, RefusalReason, SecurityLevel
from src.schemas.api import QueryRequest, QueryResponse, RetrievedDocument

# Add kREPS-rag/src to path for imports
KREPS_RAG_SRC = Path(__file__).parent.parent.parent / "kREPS-rag" / "src"
if str(KREPS_RAG_SRC) not in sys.path:
    sys.path.insert(0, str(KREPS_RAG_SRC))

logger = logging.getLogger(__name__)


# Security level mapping between API (0-4) and kREPS-rag (0-3)
# API: PUBLIC=0, INTERNAL=1, CONFIDENTIAL=2, SECRET=3, TOP_SECRET=4
# kREPS-rag: PUBLIC=0, INTERNAL=1, CONFIDENTIAL=2, RESTRICTED=3
def _api_clearance_to_kreps(api_clearance: int) -> int:
    """Map API clearance level to kREPS-rag SecurityLevel."""
    # TOP_SECRET (4) maps to RESTRICTED (3) since kREPS-rag has no TOP_SECRET
    return min(api_clearance, 3)


def _kreps_level_to_api(kreps_level_str: str) -> int:
    """Map kREPS-rag security level string to API SecurityLevel int."""
    mapping = {
        "public": SecurityLevel.PUBLIC.value,
        "internal": SecurityLevel.INTERNAL.value,
        "confidential": SecurityLevel.CONFIDENTIAL.value,
        "restricted": SecurityLevel.SECRET.value,  # RESTRICTED -> SECRET
    }
    return mapping.get(kreps_level_str.lower(), SecurityLevel.PUBLIC.value)


@dataclass
class QueryResult:
    """Internal result of query processing."""

    answer: str
    sources: list[RetrievedDocument]
    refused: bool
    refusal_reason: RefusalReason
    retrieved_count: int
    used_count: int
    top_score: float | None
    latency_ms: int


class QueryService:
    """Processes RAG queries with security filtering and audit logging."""

    @staticmethod
    async def process(request: QueryRequest, db: AsyncSession) -> QueryResponse:
        """Process a query request and return response."""
        start_time = time.perf_counter()
        query_hash_value = hash_query(request.query)

        result = await QueryService._execute_query(request)

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Create audit log
        await QueryService._create_audit_log(
            db=db,
            query_hash=query_hash_value,
            index_name=request.index_name,
            user_clearance=request.clearance_level,
            retrieved_count=result.retrieved_count,
            used_count=result.used_count,
            top_score=result.top_score,
            refused=result.refused,
            refusal_reason=result.refusal_reason,
            latency_ms=latency_ms,
        )

        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            query=request.query,
            refused=result.refused,
            refusal_reason=result.refusal_reason if result.refused else None,
        )

    @staticmethod
    async def _execute_query(request: QueryRequest) -> QueryResult:
        """Execute query with security filtering and evidence evaluation."""
        import asyncio
        from security_mapping import SecurityContext, SecurityLevel as KrepsSecurityLevel, DocumentType
        from retrieve import retrieve_chunks
        from answer import AnswerGenerator

        # Build kREPS-rag security context from API clearance level
        kreps_clearance = _api_clearance_to_kreps(request.clearance_level)
        security_context = SecurityContext(
            clearance_level=KrepsSecurityLevel(kreps_clearance),
            allowed_document_types=[
                DocumentType.POLICY,
                DocumentType.REPORT,
                DocumentType.MANUAL,
                DocumentType.OTHER,
                DocumentType.FINANCE,
                DocumentType.LEGAL,
                DocumentType.HR,
            ],
            department=None,
        )

        loop = asyncio.get_event_loop()

        # Retrieve chunks using kREPS-rag retriever (with security filtering)
        try:
            raw_results, used_context = await loop.run_in_executor(
                None,
                lambda: retrieve_chunks(request.query, k=request.top_k * 2, security_context=security_context)
            )
        except ValueError as e:
            logger.error(f"Retrieval failed: {e}")
            return QueryResult(
                answer=get_refusal_message(RefusalReason.INDEX_NOT_FOUND),
                sources=[],
                refused=True,
                refusal_reason=RefusalReason.INDEX_NOT_FOUND,
                retrieved_count=0,
                used_count=0,
                top_score=None,
                latency_ms=0,
            )

        retrieved_count = len(raw_results)
        top_score = max((r.get("score", 0.0) for r in raw_results), default=None)

        # Convert kREPS-rag chunks to API format with security level mapping
        converted_chunks: list[dict[str, Any]] = []
        for chunk in raw_results:
            metadata = chunk.get("metadata", {})
            kreps_level = metadata.get("security_level", "public")
            api_level = _kreps_level_to_api(kreps_level)

            converted_chunks.append({
                "content": chunk.get("text", ""),
                "source": metadata.get("document", "unknown"),
                "score": chunk.get("score", 0.0),
                "metadata": {
                    "page": metadata.get("page"),
                    "section": metadata.get("section", ""),
                    "security_level": api_level,
                    "language": metadata.get("language", "en"),
                },
            })

        # Additional API-level security filtering (defense in depth)
        allowed_chunks = filter_chunks_by_clearance(converted_chunks, request.clearance_level)

        if not allowed_chunks:
            return QueryResult(
                answer=get_refusal_message(RefusalReason.NO_ALLOWED_CHUNKS),
                sources=[],
                refused=True,
                refusal_reason=RefusalReason.NO_ALLOWED_CHUNKS,
                retrieved_count=retrieved_count,
                used_count=0,
                top_score=top_score,
                latency_ms=0,
            )

        # Evidence evaluation
        has_evidence, refusal_reason, relevant_chunks = evaluate_evidence(allowed_chunks)

        if not has_evidence:
            return QueryResult(
                answer=get_refusal_message(refusal_reason),
                sources=[],
                refused=True,
                refusal_reason=refusal_reason,
                retrieved_count=retrieved_count,
                used_count=0,
                top_score=top_score,
                latency_ms=0,
            )

        # Select top_k chunks for answer generation
        final_chunks = sorted(
            relevant_chunks, key=lambda x: x.get("score", 0.0), reverse=True
        )[: request.top_k]

        # Generate answer using kREPS-rag AnswerGenerator (Ollama LLM)
        try:
            generator = AnswerGenerator()

            # Build chunks in kREPS-rag format for generation
            kreps_chunks = []
            for chunk in final_chunks:
                metadata = chunk.get("metadata", {})
                kreps_chunks.append({
                    "text": chunk["content"],
                    "chunk_id": f"chunk_{hash(chunk['content'])[:8]}",
                    "score": chunk.get("score", 0.0),
                    "metadata": {
                        "document": chunk.get("source", "unknown"),
                        "page": metadata.get("page"),
                        "section": metadata.get("section", ""),
                        "security_level": "internal",  # Already filtered
                        "language": metadata.get("language", "en"),
                        "allowed_for_answer": True,
                    },
                })

            # Generate answer via Ollama
            answer_result = await loop.run_in_executor(
                None,
                lambda: generator.generate_answer(request.query, security_context)
            )

            answer = answer_result.get("answer", "Unable to generate answer.")
            confidence = answer_result.get("confidence", "Low")

            # Check if LLM refused due to insufficient evidence
            if confidence == "Low" and "Insufficient evidence" in answer:
                return QueryResult(
                    answer=get_refusal_message(RefusalReason.INSUFFICIENT_EVIDENCE),
                    sources=[],
                    refused=True,
                    refusal_reason=RefusalReason.INSUFFICIENT_EVIDENCE,
                    retrieved_count=retrieved_count,
                    used_count=len(final_chunks),
                    top_score=top_score,
                    latency_ms=0,
                )

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Fallback: provide sources without LLM answer
            answer = (
                f"Answer generation temporarily unavailable. "
                f"Found {len(final_chunks)} relevant document(s). "
                f"Error: {str(e)}"
            )

        # Build sources - strip security_level from metadata
        sources = [
            RetrievedDocument(
                content=chunk["content"],
                source=chunk.get("source", "unknown"),
                score=chunk.get("score", 0.0),
                metadata={
                    k: v
                    for k, v in chunk.get("metadata", {}).items()
                    if k != "security_level"
                },
            )
            for chunk in final_chunks
        ]

        return QueryResult(
            answer=answer,
            sources=sources,
            refused=False,
            refusal_reason=RefusalReason.NONE,
            retrieved_count=retrieved_count,
            used_count=len(final_chunks),
            top_score=top_score,
            latency_ms=0,
        )

    @staticmethod
    async def _create_audit_log(
        db: AsyncSession,
        query_hash: str,
        index_name: str,
        user_clearance: int,
        retrieved_count: int,
        used_count: int,
        top_score: float | None,
        refused: bool,
        refusal_reason: RefusalReason,
        latency_ms: int,
    ) -> None:
        """Persist audit log entry."""
        audit_log = QueryAuditLog(
            id=str(uuid4()),
            query_hash=query_hash,
            index_name=index_name,
            user_clearance_level=user_clearance,
            retrieved_count=retrieved_count,
            used_count=used_count,
            top_score=top_score,
            refused=refused,
            refusal_reason=refusal_reason,
            latency_ms=latency_ms,
        )
        db.add(audit_log)
        await db.commit()
