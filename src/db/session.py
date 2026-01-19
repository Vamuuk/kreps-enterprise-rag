"""Database session management."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, StaticPool

from src.core.config import settings


def _is_testing() -> bool:
    """Detect if running under pytest."""
    return os.environ.get("TESTING", "0") == "1"


def _create_engine() -> AsyncEngine:
    """Create appropriate engine for environment."""
    if _is_testing():
        # SQLite async in-memory for tests
        return create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False,
        )
    else:
        db_url = settings.DATABASE_URL

        # SQLite needs special handling
        if db_url.startswith("sqlite"):
            return create_async_engine(
                db_url,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
            # MySQL/PostgreSQL
            return create_async_engine(
                db_url,
                poolclass=NullPool,
                echo=False,
            )


engine: AsyncEngine = _create_engine()

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for background tasks."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Create database tables."""
    from src.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()
