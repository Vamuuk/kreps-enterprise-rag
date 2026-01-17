"""Security functions - clearance filtering and query hashing."""

import hashlib
from typing import Any

from src.db.models import SecurityLevel


def hash_query(query: str) -> str:
    """Generate deterministic SHA-256 hash of query text for audit logging."""
    normalized = query.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def filter_chunks_by_clearance(
    chunks: list[dict[str, Any]],
    user_clearance: int,
) -> list[dict[str, Any]]:
    """
    Filter chunks based on user's security clearance level.

    Access rule: chunk.security_level <= user_clearance
    Chunks above clearance are removed entirely - no leakage.
    """
    allowed = []
    for chunk in chunks:
        chunk_level = chunk.get("metadata", {}).get(
            "security_level", SecurityLevel.PUBLIC.value
        )
        if chunk_level <= user_clearance:
            allowed.append(chunk)
    return allowed
