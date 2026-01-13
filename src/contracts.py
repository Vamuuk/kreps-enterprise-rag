"""
Enterprise contract interface for offline RAG system.
Defines strict input/output schemas for query execution.
This file is the ONLY entrypoint that the frontend is allowed to call.
"""

from typing import TypedDict, Literal, List


class SourceDict(TypedDict):
    document: str
    page: int
    section: str
    score: float


class ChunkDict(TypedDict):
    chunk_id: str
    text: str
    metadata: dict
    score: float


class QueryResult(TypedDict):
    answer: str
    confidence: Literal["High", "Medium", "Low"]
    sources: List[SourceDict]
    chunks: List[ChunkDict]


def run_query(query: str) -> QueryResult:
    """
    Execute RAG query via the real backend pipeline.
    Thin contract layer – no logic here.
    """
    from src.answer import answer_query
    return answer_query(query)
