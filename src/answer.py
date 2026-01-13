"""
Answer generation module with enterprise guardrails.
Implements quality checks, confidence scoring, and answer generation via Ollama.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import List, Dict, Literal

from src.config import MIN_SOURCES, CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_MEDIUM
from src.retrieve import retrieve_chunks
from src.contracts import QueryResult, SourceDict, ChunkDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Answer generation with quality guardrails and Ollama LLM integration."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        timeout: int = 450
    ):
        """
        Initialize answer generator with Ollama connection.

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            model: Generation model name (default: qwen3:8b)
            timeout: Request timeout in seconds (default: 450)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout

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
            chunks = retrieve_chunks(query, k=5)
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

        # Generate answer using Qwen LLM
        try:
            answer = self._generate_answer_with_llm(query, chunks, confidence)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._insufficient_evidence_response(
                f"Answer generation failed: {e}"
            )

        # Post-processing: Validate answer quality
        if not self._validate_answer(answer):
            logger.warning("Answer validation failed - insufficient evidence detected")
            return self._insufficient_evidence_response(
                "The generated answer did not meet enterprise quality standards."
            )

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "chunks": formatted_chunks
        }

    def _passes_guardrails(self, chunks: List[Dict]) -> bool:
        """
        STRICT enterprise guardrails: refuse to answer if evidence is insufficient.

        Args:
            chunks: Retrieved chunks with scores

        Returns:
            True if guardrails pass, False otherwise
        """
        # Guardrail 1: Minimum number of sources (enterprise requirement)
        if len(chunks) < MIN_SOURCES:
            logger.warning(
                f"GUARDRAIL FAILED: Insufficient sources "
                f"({len(chunks)} < {MIN_SOURCES} required)"
            )
            return False

        # Guardrail 2: Minimum relevance score
        # Only answer if top chunks have meaningful relevance
        avg_score = sum(c["score"] for c in chunks[:3]) / min(3, len(chunks))
        if avg_score < 0.3:
            logger.warning(
                f"GUARDRAIL FAILED: Low relevance scores (avg={avg_score:.3f})"
            )
            return False

        # Guardrail 3: Top chunk must be significantly relevant
        if chunks[0]["score"] < 0.4:
            logger.warning(
                f"GUARDRAIL FAILED: Top chunk score too low "
                f"({chunks[0]['score']:.3f})"
            )
            return False

        logger.info("✓ All guardrails passed")
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

    def _build_prompt(self, query: str, chunks: List[Dict]) -> str:
        """
        Build STRICT enterprise prompt that enforces evidence-based answers only.

        Args:
            query: User query
            chunks: Retrieved context chunks

        Returns:
            Formatted prompt string
        """
        # STRICT system instruction
        system_msg = (
            "You are an enterprise AI assistant. "
            "You MUST answer STRICTLY and ONLY from the provided context. "
            "DO NOT use external knowledge. DO NOT make assumptions. "
            "DO NOT paraphrase beyond what the evidence explicitly states."
        )

        # Build context from top chunks
        context_parts = []
        for i, chunk in enumerate(chunks[:4], 1):  # Use top 4 chunks
            metadata = chunk["metadata"]
            doc_info = f"{metadata['document']}"
            if metadata.get('page'):
                doc_info += f", Page {metadata['page']}"
            if metadata.get('section'):
                doc_info += f", Section: {metadata['section']}"

            context_parts.append(
                f"[Source {i}: {doc_info}]\n{chunk['text']}\n"
            )

        context = "\n".join(context_parts)

        # Construct STRICT enterprise prompt
        prompt = f"""{system_msg}

QUESTION:
{query}

PROVIDED CONTEXT:
{context}

STRICT INSTRUCTIONS:
1. Answer ONLY using information explicitly stated in the provided context above
2. If the context does not contain sufficient information, you MUST respond:
   "Insufficient evidence to answer this question from the provided documentation"
3. DO NOT use external knowledge, common sense, or general information
4. DO NOT infer, assume, or extrapolate beyond what is explicitly written
5. Cite source numbers [Source N] when referencing information
6. Be precise and concise - do not add unnecessary elaboration
7. If uncertain about ANY part of the answer, default to "Insufficient evidence"

ANSWER:"""

        return prompt

    def _generate_answer_with_llm(
        self,
        query: str,
        chunks: List[Dict],
        confidence: str
    ) -> str:
        """
        Generate answer using Ollama Qwen LLM.

        Args:
            query: User query
            chunks: Retrieved context chunks
            confidence: Confidence level

        Returns:
            Generated answer string

        Raises:
            RuntimeError: If LLM generation fails
        """
        logger.info(f"Generating answer using {self.model}")

        # Build prompt
        prompt = self._build_prompt(query, chunks)

        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for factual responses
                "top_p": 0.9,
                "top_k": 40
            }
        }

        data = json.dumps(payload).encode('utf-8')

        # Create request
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
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

                # Extract generated text
                if "response" in response_data:
                    answer = response_data["response"].strip()
                else:
                    raise RuntimeError("Unexpected response format from Ollama API")

                logger.info("✓ Answer generated successfully")
                return answer

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
            error_msg = f"Unexpected error during generation: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _validate_answer(self, answer: str) -> bool:
        """
        Post-processing validation: detect if LLM properly refused to hallucinate.

        Args:
            answer: Generated answer text

        Returns:
            True if answer is valid, False if it appears to be hallucination
        """
        # If answer explicitly states insufficient evidence, that's valid
        insufficient_indicators = [
            "insufficient evidence",
            "cannot answer",
            "not enough information",
            "cannot determine",
            "no information",
            "does not contain"
        ]

        answer_lower = answer.lower()
        for indicator in insufficient_indicators:
            if indicator in answer_lower:
                logger.info("Answer correctly refuses with insufficient evidence")
                return True

        # If answer is too short, it might be invalid
        if len(answer.strip()) < 20:
            logger.warning("Answer too short - possible quality issue")
            return False

        # Answer appears to provide information - assume valid
        # (LLM followed instructions to answer from context)
        return True

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
