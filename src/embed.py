"""
Embedding engine for enterprise RAG system using Ollama.

Implements text vectorization using local Ollama server with bge-m3 model.
All operations are offline via HTTP API at localhost:11434.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import List

from src.config import EMBEDDING_DIM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Embedding engine using Ollama local server.

    Connects to Ollama HTTP API for text vectorization using bge-m3 model.
    All operations are synchronous and offline.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        batch_size: int = 32,
        timeout: int = 300
    ):
        """
        Initialize Ollama embedding engine.

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            model: Model name (default: bge-m3)
            batch_size: Number of texts to process per batch (default: 32)
            timeout: Request timeout in seconds (default: 300)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.embedding_dim = None  # Will be set after first embedding

        logger.info(f"EmbeddingEngine initialized with Ollama at {self.base_url}")
        logger.info(f"Using model: {self.model}")

        # Verify connection to Ollama
        self._verify_connection()

    def _verify_connection(self) -> None:
        """
        Verify Ollama server is accessible.

        Raises:
            ConnectionError: If Ollama server is not accessible
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    logger.info("✓ Ollama server connection verified")
                else:
                    logger.warning(f"Ollama server returned status {response.status}")
        except urllib.error.URLError as e:
            error_msg = (
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running: {e}"
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg)
        except Exception as e:
            logger.error(f"Unexpected error connecting to Ollama: {e}")
            raise

    def _call_embeddings_api(self, texts: List[str]) -> List[List[float]]:
        """
        Call Ollama embeddings API for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If API call fails
        """
        # Prepare request payload
        payload = {
            "model": self.model,
            "input": texts
        }

        data = json.dumps(payload).encode('utf-8')

        # Create request
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=data,
            headers={
                'Content-Type': 'application/json'
            },
            method='POST'
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status != 200:
                    error_msg = f"Ollama API returned status {response.status}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                # Parse response
                response_data = json.loads(response.read().decode('utf-8'))

                # Extract embeddings
                if "embeddings" in response_data:
                    embeddings = response_data["embeddings"]
                elif "embedding" in response_data:
                    # Single embedding case
                    embeddings = [response_data["embedding"]]
                else:
                    raise RuntimeError("Unexpected response format from Ollama API")

                # Validate embedding dimensions
                if embeddings and not self.embedding_dim:
                    self.embedding_dim = len(embeddings[0])
                    logger.info(f"Embedding dimension detected: {self.embedding_dim}")

                return embeddings

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else 'No error details'
            error_msg = f"Ollama API HTTP error {e.code}: {error_body}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        except urllib.error.URLError as e:
            error_msg = f"Cannot reach Ollama server: {e.reason}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from Ollama: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error during embedding: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert texts to embedding vectors using Ollama.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each vector is list of floats)

        Raises:
            RuntimeError: If embedding fails
            ValueError: If texts is empty
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")

        logger.info(f"Embedding {len(texts)} texts using {self.model}")

        all_embeddings = []

        # Process in batches for efficiency
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

            try:
                batch_embeddings = self._call_embeddings_api(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to embed batch {batch_num}: {e}")
                raise

        # Validate output
        if len(all_embeddings) != len(texts):
            error_msg = (
                f"Embedding count mismatch: expected {len(texts)}, "
                f"got {len(all_embeddings)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"✓ Successfully embedded {len(texts)} texts")
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string using Ollama.

        Args:
            query: Query string

        Returns:
            Single embedding vector

        Raises:
            RuntimeError: If embedding fails
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("Cannot embed empty query")

        embeddings = self.embed_texts([query])
        return embeddings[0]


def get_embedding_engine() -> EmbeddingEngine:
    """
    Factory function to get embedding engine instance.

    Returns:
        EmbeddingEngine instance configured for Ollama

    Raises:
        ConnectionError: If Ollama server is not accessible
    """
    return EmbeddingEngine()


if __name__ == "__main__":
    # Test embedding engine
    engine = get_embedding_engine()
    print(f"Embedding engine initialized")
    print(f"Model: {engine.model}")
    print(f"Base URL: {engine.base_url}")

    # Test single embedding
    test_text = "This is a test document for embedding."
    print(f"\nTesting single embedding...")
    try:
        embedding = engine.embed_query(test_text)
        print(f"✓ Embedding successful")
        print(f"  Dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"✗ Embedding failed: {e}")

    # Test batch embedding
    test_texts = [
        "First document about safety procedures.",
        "Second document about equipment maintenance.",
        "Third document about operational guidelines."
    ]
    print(f"\nTesting batch embedding ({len(test_texts)} texts)...")
    try:
        embeddings = engine.embed_texts(test_texts)
        print(f"✓ Batch embedding successful")
        print(f"  Number of embeddings: {len(embeddings)}")
        print(f"  Dimension: {len(embeddings[0])}")
    except Exception as e:
        print(f"✗ Batch embedding failed: {e}")
