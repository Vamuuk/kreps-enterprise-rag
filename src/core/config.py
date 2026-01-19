"""Application configuration and constants."""

import os
from pathlib import Path


class Settings:
    """Application settings. Uses environment variables or sensible defaults."""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    KREPS_RAG_ROOT: Path = PROJECT_ROOT / "kREPS-rag"
    STORAGE_DIR: Path = KREPS_RAG_ROOT / "storage"
    DATA_DIR: Path = KREPS_RAG_ROOT / "data"

    # Database - use SQLite by default for development
    # Set DATABASE_URL env var for MySQL in production
    @property
    def DATABASE_URL(self) -> str:
        env_url = os.environ.get("DATABASE_URL")
        if env_url:
            return env_url
        # Default to SQLite
        db_path = self.PROJECT_ROOT / "data" / "rag.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{db_path}"

    # API
    API_VERSION: str = "1.0.0"
    API_TITLE: str = "Enterprise RAG API"

    # Retrieval thresholds
    RELEVANCE_THRESHOLD: float = 0.3
    MIN_EVIDENCE_CHUNKS: int = 1

    # Pipeline stage progress weights (start%, end%)
    STAGE_WEIGHTS: dict[str, tuple[float, float]] = {
        "queued": (0.0, 0.0),
        "ingest": (0.0, 20.0),
        "chunk": (20.0, 40.0),
        "embed": (40.0, 80.0),
        "index": (80.0, 95.0),
        "finalize": (95.0, 100.0),
    }


settings = Settings()
