"""
Hybrid retrieval module combining semantic (FAISS) and lexical (BM25) search.
Implements score normalization and fusion for optimal results.
"""

import logging
from typing import List, Dict

from config import SEMANTIC_WEIGHT, LEXICAL_WEIGHT
from index_faiss import load_faiss_index
from index_bm25 import load_bm25_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining semantic and lexical search."""

    def __init__(self):
        """Initialize retriever with both indices."""
        logger.info("Loading indices for hybrid retrieval")
        try:
            self.faiss_index = load_faiss_index()
            self.bm25_index = load_bm25_index()
            logger.info("Indices loaded successfully")
        except ValueError as e:
            logger.error(f"Failed to load indices: {e}")
            raise

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve relevant chunks using hybrid search.

        Args:
            query: User query string
            k: Number of final results to return

        Returns:
            List of chunk dicts with scores, sorted by relevance

        Note:
            If embedding engine is not implemented, will fall back to BM25 only
        """
        # Try semantic search
        try:
            faiss_results = self.faiss_index.search(query, k=k)
            semantic_available = True
        except NotImplementedError:
            logger.warning("Semantic search unavailable (Qwen not integrated)")
            logger.warning("Falling back to BM25 only")
            faiss_results = []
            semantic_available = False

        # Lexical search (always available)
        bm25_results = self.bm25_index.search(query, k=k)

        if not semantic_available:
            # BM25 only
            return self._format_results(bm25_results, k)

        # Hybrid fusion
        merged = self._merge_results(faiss_results, bm25_results)

        # Sort by final score and take top k
        merged.sort(key=lambda x: x["score"], reverse=True)

        return merged[:k]

    def _merge_results(
        self,
        faiss_results: List[tuple],
        bm25_results: List[tuple]
    ) -> List[Dict]:
        """
        Merge and normalize scores from both retrievers.

        Args:
            faiss_results: List of (chunk, score) from FAISS
            bm25_results: List of (chunk, score) from BM25

        Returns:
            List of chunks with normalized hybrid scores
        """
        # Normalize FAISS scores
        faiss_scores = {}
        if faiss_results:
            max_faiss = max(score for _, score in faiss_results)
            min_faiss = min(score for _, score in faiss_results)
            range_faiss = max_faiss - min_faiss if max_faiss > min_faiss else 1.0

            for chunk, score in faiss_results:
                chunk_id = chunk["chunk_id"]
                normalized = (score - min_faiss) / range_faiss
                faiss_scores[chunk_id] = normalized

        # Normalize BM25 scores
        bm25_scores = {}
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results)
            min_bm25 = min(score for _, score in bm25_results)
            range_bm25 = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0

            for chunk, score in bm25_results:
                chunk_id = chunk["chunk_id"]
                normalized = (score - min_bm25) / range_bm25
                bm25_scores[chunk_id] = normalized

        # Merge results
        all_chunk_ids = set(faiss_scores.keys()) | set(bm25_scores.keys())

        # Build chunk lookup
        chunk_lookup = {}
        for chunk, _ in faiss_results:
            chunk_lookup[chunk["chunk_id"]] = chunk
        for chunk, _ in bm25_results:
            chunk_lookup[chunk["chunk_id"]] = chunk

        # Calculate hybrid scores
        merged = []
        for chunk_id in all_chunk_ids:
            semantic_score = faiss_scores.get(chunk_id, 0.0)
            lexical_score = bm25_scores.get(chunk_id, 0.0)

            # Weighted combination
            hybrid_score = (
                SEMANTIC_WEIGHT * semantic_score +
                LEXICAL_WEIGHT * lexical_score
            )

            chunk = chunk_lookup[chunk_id].copy()
            chunk["score"] = hybrid_score

            merged.append(chunk)

        return merged

    def _format_results(
        self,
        results: List[tuple],
        k: int
    ) -> List[Dict]:
        """
        Format retrieval results for output.

        Args:
            results: List of (chunk, score) tuples
            k: Number of results to return

        Returns:
            List of chunk dicts with scores
        """
        formatted = []
        for chunk, score in results[:k]:
            chunk_copy = chunk.copy()
            chunk_copy["score"] = score
            formatted.append(chunk_copy)

        return formatted


def retrieve_chunks(query: str, k: int = 10) -> List[Dict]:
    """
    Retrieve relevant chunks for a query.

    Args:
        query: User query string
        k: Number of results to return

    Returns:
        List of chunk dicts with scores

    Raises:
        ValueError: If indices not found
    """
    retriever = HybridRetriever()
    return retriever.retrieve(query, k=k)


if __name__ == "__main__":
    # Test retrieval
    query = "What are the safety procedures?"
    print(f"Query: {query}")

    try:
        results = retrieve_chunks(query, k=5)
        print(f"\nRetrieved {len(results)} chunks:")
        for i, chunk in enumerate(results, 1):
            print(f"\n{i}. Score: {chunk['score']:.4f}")
            print(f"   Document: {chunk['metadata']['document']}")
            print(f"   Section: {chunk['metadata']['section']}")
            print(f"   Text: {chunk['text'][:100]}...")
    except ValueError as e:
        print(f"Error: {e}")
        print("Run indexing first: python src/app.py index")
