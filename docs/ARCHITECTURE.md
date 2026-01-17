# System Architecture

## Overview

Offline enterprise RAG system with security-aware retrieval and deterministic refusal logic.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Presentation Layer                          │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ POST /index │  │ GET /index/ │  │ POST /query │                 │
│  │             │  │   {job_id}  │  │             │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
└─────────┼────────────────┼────────────────┼────────────────────────┘
          │                │                │
┌─────────┼────────────────┼────────────────┼────────────────────────┐
│         ▼                ▼                ▼                        │
│                      Business Layer                                 │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │ Job Management  │  │ Security Filter │  │ Evidence Evaluator  │ │
│  │                 │  │                 │  │                     │ │
│  │ - Create job    │  │ - Clearance     │  │ - Relevance check   │ │
│  │ - Track progress│  │   filtering     │  │ - Refusal logic     │ │
│  │ - State machine │  │ - Access control│  │ - Determinism       │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘ │
└───────────┼────────────────────┼─────────────────────┼─────────────┘
            │                    │                     │
┌───────────┼────────────────────┼─────────────────────┼─────────────┐
│           ▼                    ▼                     ▼             │
│                         Data Layer                                  │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │   MySQL     │  │   FAISS     │  │    BM25     │  │  Ollama   │ │
│  │             │  │             │  │             │  │           │ │
│  │ - Jobs      │  │ - Vectors   │  │ - Lexical   │  │ - Embed   │ │
│  │ - Audit logs│  │ - ANN search│  │ - Full-text │  │ - Generate│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Indexing Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  INGEST  │───▶│  CHUNK   │───▶│  EMBED   │───▶│  INDEX   │───▶│ FINALIZE │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │               │
     ▼               ▼               ▼               ▼               ▼
  Load docs      Split into      Generate       Build FAISS     Persist to
  from path      chunks with     embeddings     + BM25 indices  disk, validate
                 metadata        via Ollama

     │               │               │               │               │
     └───────────────┴───────────────┴───────────────┴───────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  Progress → MySQL   │
                          │  (after each step)  │
                          └─────────────────────┘
```

**Stage Progression:**

| Stage | Progress Range | Operation |
|-------|----------------|-----------|
| QUEUED | 0% | Job created, awaiting execution |
| INGEST | 0-20% | Load documents from filesystem |
| CHUNK | 20-40% | Split documents, preserve metadata |
| EMBED | 40-80% | Generate vector embeddings |
| INDEX | 80-95% | Build search indices |
| FINALIZE | 95-100% | Persist indices, cleanup |

### Query Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Query Request                                │
│         {query, index_name, top_k, clearance_level}                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │     Hybrid Search      │
                    │                        │
                    │  FAISS (semantic)      │
                    │  BM25 (lexical)        │
                    │  → RRF merge           │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   Security Filter      │◀── BEFORE any LLM call
                    │                        │
                    │  chunk.security_level  │
                    │  ≤ user.clearance      │
                    └───────────┬────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
            [allowed_chunks]         [filtered_out]
                    │                  (discarded)
                    ▼
        ┌────────────────────────┐
        │  Evidence Evaluator    │
        │                        │
        │  - Score ≥ threshold?  │
        │  - Sufficient chunks?  │
        └───────────┬────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
   [sufficient]           [insufficient]
        │                       │
        ▼                       ▼
┌───────────────┐      ┌───────────────┐
│ LLM Generate  │      │ Deterministic │
│ (Ollama)      │      │ Refusal       │
│ temp=0        │      │               │
└───────┬───────┘      └───────┬───────┘
        │                      │
        └──────────┬───────────┘
                   ▼
        ┌────────────────────────┐
        │     Audit Log          │
        │                        │
        │  - query_hash (SHA256) │
        │  - retrieved_count     │
        │  - used_count          │
        │  - top_score           │
        │  - refused + reason    │
        │  - latency_ms          │
        └────────────────────────┘
```

---

## Security Model

### Clearance-Based Access Control

```
Document Metadata          User Request
┌─────────────────┐       ┌─────────────────┐
│ security_level: │       │ clearance_level:│
│ 2 (CONFIDENTIAL)│       │ 1 (INTERNAL)    │
└────────┬────────┘       └────────┬────────┘
         │                         │
         └──────────┬──────────────┘
                    │
                    ▼
            ┌───────────────┐
            │   2 ≤ 1 ?     │
            │   FALSE       │
            └───────┬───────┘
                    │
                    ▼
              ┌──────────┐
              │ FILTERED │
              └──────────┘
```

### Security Filtering BEFORE LLM

Critical design decision: Security filtering occurs **before** any content reaches the LLM.

```
Retrieval → Security Filter → LLM
              ▲
              │
    Restricted chunks NEVER
    reach context window
```

**Why:**
- Prevents prompt injection attacks exposing restricted data
- LLM cannot hallucinate or leak filtered content
- Audit trail captures pre-filter vs post-filter counts

---

## Stateless Design

### Why Stateless

| Concern | Stateless Solution |
|---------|-------------------|
| Worker crashes | Job state in MySQL survives |
| Horizontal scaling | Any worker serves any request |
| Load balancing | Round-robin without session affinity |
| Deployment | Rolling updates without coordination |

### State Location

| State Type | Storage | Rationale |
|------------|---------|-----------|
| Job progress | MySQL `indexing_jobs` | Survives worker restarts |
| Query audit | MySQL `query_audit_logs` | Compliance, analytics |
| Vector indices | Filesystem | Shared via NFS/S3 in production |
| In-flight requests | None | Request-scoped only |

### No In-Memory State

```python
# WRONG - stateful
class JobManager:
    def __init__(self):
        self.jobs = {}  # Lost on restart

# CORRECT - stateless
async def get_job_status(job_id: str, db: AsyncSession):
    result = await db.execute(
        select(IndexingJob).where(IndexingJob.id == job_id)
    )
    return result.scalar_one_or_none()
```

---

## Horizontal Scaling

### Scaling Points

```
                                    ┌─────────────┐
                              ┌────▶│  Worker 1   │
                              │     └─────────────┘
┌──────────┐    ┌──────────┐  │     ┌─────────────┐
│  Client  │───▶│    LB    │──┼────▶│  Worker 2   │ ◀── Stateless API
│          │    │          │  │     └─────────────┘     (scale horizontally)
└──────────┘    └──────────┘  │     ┌─────────────┐
                              └────▶│  Worker N   │
                                    └─────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
             ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
             │   MySQL     │        │    FAISS    │        │   Ollama    │
             │  (primary)  │        │  (readonly) │        │  (GPU pool) │
             └─────────────┘        └─────────────┘        └─────────────┘
                    │                                             │
                    ▼                                             ▼
             ┌─────────────┐                               ┌─────────────┐
             │   MySQL     │ ◀── Read replicas             │   Ollama    │
             │  (replica)  │     for query load            │  (replica)  │
             └─────────────┘                               └─────────────┘
```

### Scaling Recommendations

| Component | Scaling Strategy | Bottleneck |
|-----------|-----------------|------------|
| API Workers | Horizontal (add instances) | CPU-bound request handling |
| MySQL | Read replicas | Query audit writes |
| FAISS | Sharding by index | Memory per index |
| Ollama | GPU pool | Embedding/generation throughput |

---

## Deterministic Behavior

### Same Input → Same Output

| Mechanism | Implementation |
|-----------|----------------|
| Query hashing | SHA-256 of normalized query |
| LLM temperature | `temperature=0.0` |
| Retrieval scoring | Fixed ranking algorithm |
| Refusal thresholds | Configurable constants |

### Refusal Reasons

| Reason | Trigger | Message |
|--------|---------|---------|
| `insufficient_evidence` | < min_chunks above threshold | Fixed refusal text |
| `no_allowed_chunks` | All chunks filtered by clearance | Fixed refusal text |
| `below_relevance_threshold` | No chunks score ≥ threshold | Fixed refusal text |
| `index_not_found` | Index does not exist | Fixed refusal text |

---

## File Structure

```
src/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── app.py           # FastAPI application factory
│   └── routes.py        # API route handlers
├── core/
│   ├── __init__.py
│   ├── config.py        # Application settings
│   ├── refusal.py       # Deterministic refusal logic
│   └── security.py      # Security filtering, query hashing
├── db/
│   ├── __init__.py
│   ├── models.py        # ORM models (IndexingJob, QueryAuditLog)
│   └── session.py       # SQLAlchemy async engine, session factory
├── schemas/
│   ├── __init__.py
│   └── api.py           # Pydantic request/response schemas
└── services/
    ├── __init__.py
    ├── indexing.py      # Indexing pipeline orchestration
    └── query.py         # Query processing service

kREPS-rag/
├── src/
│   ├── ingest.py        # Document loading
│   ├── chunking.py      # Token-aware text chunking
│   ├── embed.py         # Ollama embedding engine
│   ├── index_faiss.py   # FAISS vector index
│   ├── index_bm25.py    # BM25 lexical index
│   ├── retrieve.py      # Hybrid search (FAISS + BM25)
│   ├── answer.py        # LLM answer generation
│   └── security_mapping.py  # Security classification
└── storage/
    ├── faiss/           # FAISS indices
    └── bm25/            # BM25 indices

tests/
├── conftest.py
└── test_integration.py

docs/
├── ARCHITECTURE.md
└── RUNBOOK.md
```
