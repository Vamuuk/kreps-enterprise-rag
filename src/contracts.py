"""
Enterprise contract interface for offline RAG system.
Defines strict input/output schemas for query execution.
"""

from typing import TypedDict, Literal


class SourceDict(TypedDict):
    """Single source reference."""
    document: str
    page: int
    section: str
    score: float


class ChunkDict(TypedDict):
    """Retrieved chunk with metadata."""
    chunk_id: str
    text: str
    metadata: dict
    score: float


class QueryResult(TypedDict):
    """Complete query result contract."""
    answer: str
    confidence: Literal["High", "Medium", "Low"]
    sources: list[SourceDict]
    chunks: list[ChunkDict]


def run_query(query: str) -> QueryResult:
    """
    Execute RAG query and return structured result.

    This is the main contract interface that the UI calls.
    All RAG logic flows through this function.

    Args:
        query: User question string

    Returns:
        QueryResult dict with answer, confidence, sources, and chunks

    Note:
        This is a TEMPORARY placeholder.
        Real implementation in answer.py will be wired here.
    """
    # TEMPORARY PLACEHOLDER - Will be replaced with real pipeline
    return {
        "answer": "System is ready but indexing has not been completed. Please run 'python src/app.py index' first.",
        "confidence": "Low",
        "sources": [],
        "chunks": []
    }
