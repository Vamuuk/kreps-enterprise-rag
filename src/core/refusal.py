"""Deterministic refusal logic - evidence evaluation and refusal messages."""

from typing import Any

from src.core.config import settings
from src.db.models import RefusalReason

# Fixed refusal messages - deterministic responses
REFUSAL_MESSAGES: dict[RefusalReason, str] = {
    RefusalReason.INSUFFICIENT_EVIDENCE: (
        "I cannot provide an answer due to insufficient evidence in the available documents."
    ),
    RefusalReason.NO_ALLOWED_CHUNKS: (
        "I cannot provide an answer as no relevant documents are accessible at your clearance level."
    ),
    RefusalReason.BELOW_RELEVANCE_THRESHOLD: (
        "I cannot provide an answer as no sufficiently relevant documents were found."
    ),
    RefusalReason.INDEX_NOT_FOUND: (
        "I cannot provide an answer as the specified index does not exist."
    ),
}


def get_refusal_message(reason: RefusalReason) -> str:
    """Get deterministic refusal message for a given reason."""
    return REFUSAL_MESSAGES.get(
        reason,
        "I cannot provide an answer at this time.",
    )


def evaluate_evidence(
    chunks: list[dict[str, Any]],
    relevance_threshold: float | None = None,
    min_chunks: int | None = None,
) -> tuple[bool, RefusalReason, list[dict[str, Any]]]:
    """
    Evaluate if retrieved chunks provide sufficient evidence.

    Returns:
        has_evidence: Whether sufficient evidence exists
        refusal_reason: Reason for refusal if insufficient
        relevant_chunks: Chunks above relevance threshold
    """
    threshold = relevance_threshold or settings.RELEVANCE_THRESHOLD
    minimum = min_chunks or settings.MIN_EVIDENCE_CHUNKS

    if not chunks:
        return False, RefusalReason.INSUFFICIENT_EVIDENCE, []

    relevant = [c for c in chunks if c.get("score", 0.0) >= threshold]

    if not relevant:
        return False, RefusalReason.BELOW_RELEVANCE_THRESHOLD, []

    if len(relevant) < minimum:
        return False, RefusalReason.INSUFFICIENT_EVIDENCE, []

    return True, RefusalReason.NONE, relevant
