"""Application configuration and constants."""


class Settings:
    """Application settings. Replace with environment variables in production."""

    # Database
    DATABASE_URL: str = "mysql+aiomysql://root:password@localhost:3306/rag_system"

    # API
    API_VERSION: str = "1.0.0"
    API_TITLE: str = "Enterprise RAG API"

    # Retrieval thresholds
    RELEVANCE_THRESHOLD: float = 0.3
    MIN_EVIDENCE_CHUNKS: int = 1

    # Pipeline stage progress weights (start%, end%)
    # Keys are JobStage enum values
    STAGE_WEIGHTS: dict[str, tuple[float, float]] = {
        "queued": (0.0, 0.0),
        "ingest": (0.0, 20.0),
        "chunk": (20.0, 40.0),
        "embed": (40.0, 80.0),
        "index": (80.0, 95.0),
        "finalize": (95.0, 100.0),
    }


settings = Settings()
