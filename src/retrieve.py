# Гибридный поиск: FAISS (семантика) + BM25 (ключевые слова)
# +15% бонус за совпадение языка запроса и документа

import logging
import re
from typing import List, Dict

from src.config import SEMANTIC_WEIGHT, LEXICAL_WEIGHT
from src.index_faiss import load_faiss_index
from src.index_bm25 import load_bm25_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageDetector:
    @staticmethod
    def detect_query_language(query: str) -> str:
        # Определяем язык запроса по кириллице/латинице
        cyrillic_count = len(re.findall(r'[а-яА-ЯёЁ]', query))
        latin_count = len(re.findall(r'[a-zA-Z]', query))

        if cyrillic_count > latin_count * 0.3:
            return "ru"
        elif latin_count > 3:
            return "en"
        else:
            return "unknown"


class HybridRetriever:
    LANGUAGE_BOOST = 0.15  # Буст за совпадение языка

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

        self.lang_detector = LanguageDetector()

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve relevant chunks using hybrid search with language awareness.

        Args:
            query: User query string
            k: Number of final results to return

        Returns:
            List of chunk dicts with scores, sorted by relevance

        Note:
            If embedding engine is not implemented, will fall back to BM25 only
        """
        # Detect query language
        query_lang = self.lang_detector.detect_query_language(query)
        logger.info(f"Query language detected: {query_lang}")

        # Try semantic search
        try:
            faiss_results = self.faiss_index.search(query, k=k * 2)  # Get more for filtering
            semantic_available = True
        except NotImplementedError:
            logger.warning("Semantic search unavailable (Qwen not integrated)")
            logger.warning("Falling back to BM25 only")
            faiss_results = []
            semantic_available = False

        # Lexical search (always available)
        bm25_results = self.bm25_index.search(query, k=k * 2)

        if not semantic_available:
            # BM25 only
            formatted = self._format_results(bm25_results, k)
            return self._apply_language_boost(formatted, query_lang, k)

        # Hybrid fusion
        merged = self._merge_results(faiss_results, bm25_results)

        # Apply language boost
        merged = self._apply_language_boost(merged, query_lang, k)

        return merged

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

    def _apply_language_boost(
        self,
        chunks: List[Dict],
        query_lang: str,
        k: int
    ) -> List[Dict]:
        """
        Boost chunks that match the query language.

        Args:
            chunks: Retrieved chunks with scores
            query_lang: Detected query language
            k: Number of results to return

        Returns:
            Top k chunks with language boost applied
        """
        if query_lang == "unknown":
            # No language boost if query language unknown
            chunks.sort(key=lambda x: x["score"], reverse=True)
            return chunks[:k]

        # Apply boost to matching language
        for chunk in chunks:
            chunk_lang = chunk.get("metadata", {}).get("language", "unknown")

            if chunk_lang == query_lang:
                # Boost score for same language
                chunk["score"] = chunk["score"] * (1.0 + self.LANGUAGE_BOOST)
                logger.debug(
                    f"Language boost applied to chunk from "
                    f"{chunk['metadata'].get('document')} (lang={chunk_lang})"
                )

        # Re-sort after boost
        chunks.sort(key=lambda x: x["score"], reverse=True)

        return chunks[:k]

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
    Retrieve relevant chunks for a query with language awareness.

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
    test_queries = [
        "What are the safety procedures?",
        "Какие процедуры безопасности?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        try:
            results = retrieve_chunks(query, k=5)
            print(f"\nRetrieved {len(results)} chunks:")
            for i, chunk in enumerate(results, 1):
                meta = chunk.get('metadata', {})
                print(f"\n{i}. Score: {chunk['score']:.4f}")
                print(f"   Document: {meta.get('document')}")
                print(f"   Language: {meta.get('language')}")
                print(f"   Type: {meta.get('document_type')}")
                print(f"   Section: {meta.get('section')}")
                print(f"   Text: {chunk['text'][:100]}...")
        except ValueError as e:
            print(f"Error: {e}")
            print("Run indexing first: python src/app.py index")
