"""
FAISS vector indexing module for semantic search.
Manages vector index creation, persistence, and retrieval.
"""

import logging
import pickle
from typing import List, Dict, Tuple

import numpy as np
import faiss

from src.config import FAISS_DIR, EMBEDDING_DIM, TOP_K_FAISS
from src.embed import get_embedding_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS vector index manager for semantic retrieval."""

    def __init__(self):
        self.index = None
        self.chunks = []
        self.embedding_engine = get_embedding_engine()
        self.index_path = FAISS_DIR / "index.faiss"
        self.chunks_path = FAISS_DIR / "chunks.pkl"

    def build_index(self, chunks: List[Dict]) -> None:
        """
        Build FAISS index from chunks.

        Args:
            chunks: List of chunk dicts with text and metadata

        Raises:
            NotImplementedError: If embedding engine is not implemented
        """
        logger.info(f"Building FAISS index for {len(chunks)} chunks")

        self.chunks = chunks

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]

        # Get embeddings from Qwen model
        # NOTE: This will raise NotImplementedError until Qwen is integrated
        logger.info("Generating embeddings (requires Qwen integration)...")
        try:
            embeddings = self.embedding_engine.embed_texts(texts)
        except NotImplementedError as e:
            logger.error(f"Cannot build index: {e}")
            logger.error("Please integrate Qwen embedding model in embed.py")
            raise

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Normalize vectors (improves cosine similarity search)
        faiss.normalize_L2(embeddings_array)

        # Create FAISS index (L2 distance, but normalized vectors = cosine similarity)
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)

        # Add vectors to index
        self.index.add(embeddings_array)

        logger.info(f"FAISS index built with {self.index.ntotal} vectors")

        # Save index
        self.save()

    def save(self) -> None:
        """Save FAISS index and chunks to disk."""
        FAISS_DIR.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Save chunks metadata
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"FAISS index saved to {FAISS_DIR}")

    def load(self) -> bool:
        """
        Load FAISS index from disk.

        Returns:
            True if successful, False if index not found
        """
        if not self.index_path.exists() or not self.chunks_path.exists():
            logger.warning("FAISS index not found")
            return False

        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))

        # Load chunks metadata
        with open(self.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
        return True

    def search(self, query: str, k: int = TOP_K_FAISS) -> List[Tuple[Dict, float]]:
        """
        Search for similar chunks using semantic similarity.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of (chunk, score) tuples, sorted by relevance

        Raises:
            NotImplementedError: If embedding engine is not implemented
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load() or build_index() first.")

        # Embed query
        try:
            query_vector = self.embedding_engine.embed_query(query)
        except NotImplementedError as e:
            logger.error(f"Cannot search: {e}")
            logger.error("Please integrate Qwen embedding model in embed.py")
            raise

        # Convert to numpy and normalize
        query_array = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_array)

        # Search
        k = min(k, self.index.ntotal)  # Don't request more than available
        distances, indices = self.index.search(query_array, k)

        # Convert distances to similarity scores (1 / (1 + distance))
        # Lower distance = higher similarity
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            chunk = self.chunks[idx]
            # Convert L2 distance to similarity score [0, 1]
            score = 1.0 / (1.0 + distance)
            results.append((chunk, score))

        return results


def build_faiss_index(chunks: List[Dict]) -> None:
    """
    Build and save FAISS index.

    Args:
        chunks: List of chunk dicts
    """
    index = FAISSIndex()
    index.build_index(chunks)


def load_faiss_index() -> FAISSIndex:
    """
    Load existing FAISS index.

    Returns:
        FAISSIndex instance

    Raises:
        ValueError: If index not found
    """
    index = FAISSIndex()
    if not index.load():
        raise ValueError("FAISS index not found. Run indexing first.")
    return index


if __name__ == "__main__":
    # Test FAISS indexing
    from src.chunking import load_chunks

    chunks = load_chunks()
    if chunks:
        print(f"Building FAISS index for {len(chunks)} chunks")
        build_faiss_index(chunks)
    else:
        print("No chunks found. Run ingestion and chunking first.")
