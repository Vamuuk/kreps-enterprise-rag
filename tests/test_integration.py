"""Integration tests for RAG API."""

import sys
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from src.api.app import app
from src.db.models import Base
from src.db.session import async_session_factory, engine

# Get mock modules from conftest (already injected into sys.modules)
mock_retrieve_module = sys.modules["retrieve"]
mock_answer_module = sys.modules["answer"]


# =============================================================================
# Fixtures
# =============================================================================
@pytest_asyncio.fixture(scope="function")
async def setup_database():
    """Initialize test database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(setup_database):
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# =============================================================================
# Mock data
# =============================================================================
MOCK_CHUNKS_PUBLIC = [
    {
        "text": "This is public information about the company.",
        "chunk_id": "chunk_001",
        "score": 0.85,
        "metadata": {
            "document": "public_docs.pdf",
            "page": 1,
            "section": "Introduction",
            "security_level": "public",
            "language": "en",
            "allowed_for_answer": True,
        },
    },
    {
        "text": "Company policies and guidelines.",
        "chunk_id": "chunk_002",
        "score": 0.78,
        "metadata": {
            "document": "policy_guide.pdf",
            "page": 5,
            "section": "Policies",
            "security_level": "internal",
            "language": "en",
            "allowed_for_answer": True,
        },
    },
]

MOCK_ANSWER_RESULT = {
    "answer": "Based on the available documents, here is the answer to your question.",
    "confidence": "High",
    "sources": [],
    "chunks": [],
}


# =============================================================================
# Helper to configure mocks per test
# =============================================================================
def setup_successful_retrieval():
    """Configure mocks for successful retrieval."""
    mock_context = MagicMock()
    mock_context.clearance_level.name = "INTERNAL"
    mock_retrieve_module.retrieve_chunks.return_value = (MOCK_CHUNKS_PUBLIC, mock_context)
    mock_retrieve_module.retrieve_chunks.side_effect = None

    mock_generator_instance = MagicMock()
    mock_generator_instance.generate_answer.return_value = MOCK_ANSWER_RESULT
    mock_answer_module.AnswerGenerator.return_value = mock_generator_instance


def setup_index_not_found():
    """Configure mocks for index not found scenario."""
    mock_retrieve_module.retrieve_chunks.side_effect = ValueError("Index not found")


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.asyncio
async def test_query_refuses_when_no_index(client: AsyncClient):
    """Query returns refusal when index doesn't exist."""
    setup_index_not_found()

    response = await client.post(
        "/query",
        json={
            "query": "What is the policy?",
            "index_name": "nonexistent",
            "top_k": 5,
            "clearance_level": 0,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["refused"] is True
    assert data["sources"] == []
    assert "cannot provide an answer" in data["answer"].lower()


@pytest.mark.asyncio
async def test_query_filters_restricted_content(client: AsyncClient):
    """Query filters out content above user clearance."""
    setup_successful_retrieval()

    response = await client.post(
        "/query",
        json={
            "query": "Strategic plans",
            "index_name": "test_index",
            "top_k": 5,
            "clearance_level": 0,  # PUBLIC only
        },
    )

    assert response.status_code == 200
    data = response.json()

    # security_level is never exposed to client
    for source in data.get("sources", []):
        assert "security_level" not in source.get("metadata", {})


@pytest.mark.asyncio
async def test_query_succeeds_with_evidence(client: AsyncClient):
    """Query succeeds when sufficient evidence exists."""
    setup_successful_retrieval()

    response = await client.post(
        "/query",
        json={
            "query": "Company information",
            "index_name": "test_index",
            "top_k": 5,
            "clearance_level": 2,  # CONFIDENTIAL
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["refused"] is False
    assert data["refusal_reason"] is None
    assert len(data["sources"]) > 0
    assert data["query"] == "Company information"

    for source in data["sources"]:
        assert "content" in source
        assert "source" in source
        assert "score" in source


@pytest.mark.asyncio
async def test_audit_log_created(client: AsyncClient, setup_database):
    """Audit log is created for each query."""
    setup_index_not_found()

    await client.post(
        "/query",
        json={
            "query": "Test query",
            "index_name": "audit_test",
            "top_k": 5,
            "clearance_level": 1,
        },
    )

    async with async_session_factory() as session:
        result = await session.execute(
            text("SELECT COUNT(*) FROM query_audit_logs")
        )
        count = result.scalar()
        assert count >= 1

        result = await session.execute(
            text("""
                SELECT query_hash, index_name, user_clearance_level,
                       retrieved_count, latency_ms
                FROM query_audit_logs
                ORDER BY created_at DESC LIMIT 1
            """)
        )
        row = result.fetchone()
        assert row is not None
        assert len(row[0]) == 64  # SHA-256 hash
        assert row[1] == "audit_test"
        assert row[2] == 1


@pytest.mark.asyncio
async def test_deterministic_response(client: AsyncClient):
    """Same query produces same response."""
    setup_index_not_found()

    payload = {
        "query": "Deterministic test",
        "index_name": "test",
        "top_k": 5,
        "clearance_level": 2,
    }

    r1 = await client.post("/query", json=payload)
    r2 = await client.post("/query", json=payload)

    d1, d2 = r1.json(), r2.json()

    assert d1["answer"] == d2["answer"]
    assert d1["refused"] == d2["refused"]
    assert d1["refusal_reason"] == d2["refusal_reason"]


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient, setup_database):
    """Health endpoint reports status."""
    response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["database"] in ["healthy", "unhealthy"]
    assert "version" in data


@pytest.mark.asyncio
async def test_query_with_internal_clearance(client: AsyncClient):
    """User with INTERNAL clearance can access internal docs."""
    setup_successful_retrieval()

    response = await client.post(
        "/query",
        json={
            "query": "Internal policies",
            "index_name": "test_index",
            "top_k": 5,
            "clearance_level": 1,  # INTERNAL
        },
    )

    assert response.status_code == 200
    data = response.json()
    # With INTERNAL clearance, both public and internal docs are accessible
    assert data["refused"] is False
