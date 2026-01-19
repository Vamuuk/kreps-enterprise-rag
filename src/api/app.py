"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.core.config import settings
from src.db.session import close_db, init_db
from src.schemas.api import ErrorResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    await init_db()
    logger.info("Database initialized")
    yield
    await close_db()
    logger.info("Database closed")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.API_TITLE,
        description="Offline enterprise RAG system with job-based indexing",
        version=settings.API_VERSION,
        lifespan=lifespan,
        responses={
            500: {"model": ErrorResponse, "description": "Internal server error"},
        },
    )

    # Mount all routes under /api prefix
    app.include_router(router, prefix="/api")

    return app


# Application instance for uvicorn
app = create_app()
