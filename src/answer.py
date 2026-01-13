"""
Answer generation module with enterprise guardrails.
Implements quality checks and confidence scoring WITHOUT LLM.

LLM INTEGRATION POINT:
When Qwen LLM is integrated, replace generate_placeholder_answer()
with actual LLM inference using the retrieved context.
"""

import logging
from typing import List, Dict, Literal

from config import MIN_SOURCES, CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_MEDIUM
from retrieve import retrieve_chunks
from contracts import QueryResult, SourceDict, ChunkDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Answer generation with quality guardrails."""

    def __init__(self):
        """Initialize answer generator."""
        pass

    def generate_answer(self, query: str) -> QueryResult:
        """
        Generate answer for query with guardrails.

        Args:
            query: User query string

        Returns:
            QueryResult dict with answer, confidence, sources, and chunks
        """
        logger.info(f"Processing query: {query}")

        # Retrieve relevant chunks
        try:
            chunks = retrieve_chunks(query, k=10)
        except ValueError as e:
            logger.error(f"Retrieval failed: {e}")
            return self._insufficient_evidence_response(
                "Index not found. Please run indexing first."
            )

        if not chunks:
            return self._insufficient_evidence_response(
                "No relevant information found in the knowledge base."
            )

        # Apply guardrails
        if not self._passes_guardrails(chunks):
            return self._insufficient_evidence_response(
                "Insufficient evidence to provide a reliable answer."
            )

        # Determine confidence level
        confidence = self._calculate_confidence(chunks)

        # Extract sources
        sources = self._extract_sources(chunks)

        # Format chunks for output
        formatted_chunks = self._format_chunks(chunks)

        # Generate answer
        # INTEGRATION POINT: Replace with Qwen LLM inference
        answer = self._generate_placeholder_answer(query, chunks, confidence)

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "chunks": formatted_chunks
        }

    def _passes_guardrails(self, chunks: List[Dict]) -> bool:
        """
        Check if retrieval results pass quality thresholds.

        Args:
            chunks: Retrieved chunks with scores

        Returns:
            True if guardrails pass, False otherwise
        """
        if len(chunks) < MIN_SOURCES:
            logger.warning(f"Insufficient sources: {len(chunks)} < {MIN_SOURCES}")
            return False

        # Check if top score is reasonable
        avg_score = sum(c["score"] for c in chunks[:3]) / min(3, len(chunks))
        if avg_score < 0.3:  # Arbitrary threshold for "relevance"
            logger.warning(f"Low relevance scores: avg={avg_score:.3f}")
            return False

        return True

    def _calculate_confidence(self, chunks: List[Dict]) -> Literal["High", "Medium", "Low"]:
        """
        Calculate confidence level from retrieval scores.

        Args:
            chunks: Retrieved chunks with scores

        Returns:
            Confidence level: High, Medium, or Low
        """
        # Use top-3 average score as confidence indicator
        top_scores = [c["score"] for c in chunks[:3]]
        avg_score = sum(top_scores) / len(top_scores)

        if avg_score >= CONFIDENCE_THRESHOLD_HIGH:
            return "High"
        elif avg_score >= CONFIDENCE_THRESHOLD_MEDIUM:
            return "Medium"
        else:
            return "Low"

    def _extract_sources(self, chunks: List[Dict]) -> List[SourceDict]:
        """
        Extract unique sources from chunks.

        Args:
            chunks: Retrieved chunks

        Returns:
            List of source dicts
        """
        sources = []
        seen = set()

        for chunk in chunks:
            metadata = chunk["metadata"]
            key = (metadata["document"], metadata["page"], metadata["section"])

            if key not in seen:
                seen.add(key)
                sources.append({
                    "document": metadata["document"],
                    "page": metadata["page"] or 0,
                    "section": metadata["section"],
                    "score": chunk["score"]
                })

        return sources

    def _format_chunks(self, chunks: List[Dict]) -> List[ChunkDict]:
        """
        Format chunks for contract output.

        Args:
            chunks: Retrieved chunks

        Returns:
            List of formatted chunk dicts
        """
        formatted = []
        for chunk in chunks:
            formatted.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": chunk["score"]
            })
        return formatted

    def _generate_placeholder_answer(
        self,
        query: str,
        chunks: List[Dict],
        confidence: str
    ) -> str:
        """
        Generate placeholder answer WITHOUT LLM.

        INTEGRATION POINT: Replace this entire function with Qwen LLM inference.

        Args:
            query: User query
            chunks: Retrieved context chunks
            confidence: Confidence level

        Returns:
            Placeholder answer string

        TODO: When integrating Qwen LLM:
            1. Construct prompt with query and retrieved chunks
            2. Call Qwen model for inference
            3. Extract and clean generated answer
            4. Return answer

        Example implementation:
            prompt = self._build_prompt(query, chunks)
            llm_output = qwen_model.generate(prompt)
            answer = self._extract_answer(llm_output)
            return answer
        """
        # For now, return a structured placeholder that shows system is working
        top_docs = list(set(c["metadata"]["document"] for c in chunks[:3]))

        answer_parts = [
            f"[PLACEHOLDER ANSWER - Qwen LLM not yet integrated]",
            f"",
            f"Query: {query}",
            f"Confidence: {confidence}",
            f"",
            f"Retrieved information from {len(chunks)} relevant chunks across {len(top_docs)} documents:",
        ]

        for doc in top_docs[:3]:
            answer_parts.append(f"  • {doc}")

        answer_parts.extend([
            f"",
            f"To generate actual answers, integrate Qwen LLM in answer.py:",
            f"  1. Load Qwen model (offline)",
            f"  2. Construct prompt with query + retrieved context",
            f"  3. Run inference",
            f"  4. Return generated answer",
            f"",
            f"Retrieval system is working correctly. Chunks are available in the 'chunks' field."
        ])

        return "\n".join(answer_parts)

    def _insufficient_evidence_response(self, message: str) -> QueryResult:
        """
        Return insufficient evidence response.

        Args:
            message: Explanation message

        Returns:
            QueryResult with low confidence
        """
        return {
            "answer": f"Insufficient evidence to provide a reliable answer.\n\n{message}",
            "confidence": "Low",
            "sources": [],
            "chunks": []
        }


def answer_query(query: str) -> QueryResult:
    """
    Main entry point for query answering.

    Args:
        query: User query string

    Returns:
        QueryResult dict
    """
    generator = AnswerGenerator()
    return generator.generate_answer(query)


if __name__ == "__main__":
    # Test answer generation
    query = "What are the safety procedures?"
    result = answer_query(query)

    print("=" * 60)
    print("QUERY RESULT")
    print("=" * 60)
    print(f"\nQuery: {query}")
    print(f"\nConfidence: {result['confidence']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {len(result['sources'])}")
    print(f"Chunks: {len(result['chunks'])}")
