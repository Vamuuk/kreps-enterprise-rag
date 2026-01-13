"""
BM25 lexical indexing module for keyword-based search.
Provides traditional information retrieval to complement semantic search.
"""

import logging
import pickle
from typing import List, Dict, Tuple

from rank_bm25 import BM25Okapi

from src.config import BM25_DIR, TOP_K_BM25

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 lexical index manager for keyword retrieval."""

    def __init__(self):
        self.index = None
        self.chunks = []
        self.tokenized_corpus = []
        self.index_path = BM25_DIR / "bm25.pkl"
        self.chunks_path = BM25_DIR / "chunks.pkl"

    def build_index(self, chunks: List[Dict]) -> None:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dicts with text and metadata
        """
        logger.info(f"Building BM25 index for {len(chunks)} chunks")

        self.chunks = chunks

        # Tokenize corpus
        self.tokenized_corpus = [
            self._tokenize(chunk["text"]) for chunk in chunks
        ]

        # Build BM25 index
        self.index = BM25Okapi(self.tokenized_corpus)

        logger.info("BM25 index built")

        # Save index
        self.save()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Simple whitespace + lowercase tokenization
        # Can be improved with stemming/lemmatization if needed
        return text.lower().split()

    def save(self) -> None:
        """Save BM25 index and chunks to disk."""
        BM25_DIR.mkdir(parents=True, exist_ok=True)

        # Save BM25 index
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "index": self.index,
                "tokenized_corpus": self.tokenized_corpus
            }, f)

        # Save chunks metadata
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"BM25 index saved to {BM25_DIR}")

    def load(self) -> bool:
        """
        Load BM25 index from disk.

        Returns:
            True if successful, False if index not found
        """
        if not self.index_path.exists() or not self.chunks_path.exists():
            logger.warning("BM25 index not found")
            return False

        # Load BM25 index
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.index = data["index"]
            self.tokenized_corpus = data["tokenized_corpus"]

        # Load chunks metadata
        with open(self.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(f"BM25 index loaded: {len(self.chunks)} chunks")
        return True

    def search(self, query: str, k: int = TOP_K_BM25) -> List[Tuple[Dict, float]]:
        """
        Search for relevant chunks using BM25 scoring.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of (chunk, score) tuples, sorted by relevance
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load() or build_index() first.")

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self.index.get_scores(query_tokens)

        # Get top-k indices
        k = min(k, len(scores))
        top_indices = scores.argsort()[-k:][::-1]

        # Build results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(scores[idx])

            # Only include results with positive scores
            if score > 0:
                results.append((chunk, score))

        return results


def build_bm25_index(chunks: List[Dict]) -> None:
    """
    Build and save BM25 index.

    Args:
        chunks: List of chunk dicts
    """
    index = BM25Index()
    index.build_index(chunks)


def load_bm25_index() -> BM25Index:
    """
    Load existing BM25 index.

    Returns:
        BM25Index instance

    Raises:
        ValueError: If index not found
    """
    index = BM25Index()
    if not index.load():
        raise ValueError("BM25 index not found. Run indexing first.")
    return index


if __name__ == "__main__":
    # Test BM25 indexing
    from src.chunking import load_chunks

    chunks = load_chunks()
    if chunks:
        print(f"Building BM25 index for {len(chunks)} chunks")
        build_bm25_index(chunks)
    else:
        print("No chunks found. Run ingestion and chunking first.")
