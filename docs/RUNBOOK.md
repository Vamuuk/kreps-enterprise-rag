# Operational Runbook

## Prerequisites

- Python 3.11+
- MySQL 8.0+
- Ollama (local LLM runtime)

---

## 1. Start MySQL

```bash
# Linux/macOS
sudo systemctl start mysql

# Windows
net start MySQL80

# Docker alternative
docker run -d \
  --name rag-mysql \
  -e MYSQL_ROOT_PASSWORD=password \
  -e MYSQL_DATABASE=rag_system \
  -p 3306:3306 \
  mysql:8.0
```

Verify connection:

```bash
mysql -u root -p -e "SELECT 1"
```

---

## 2. Start Ollama

```bash
# Start Ollama service
ollama serve

# Pull required models (run once)
ollama pull nomic-embed-text
ollama pull llama3.2
```

Verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

---

## 3. Start FastAPI

### Development (Single Worker)

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r kREPS-rag/requirements.txt

# Start with hot reload
python run.py
# Or directly:
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Production (Multiple Workers)

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Verify API is running:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "healthy", "database": "healthy", "version": "1.0.0"}
```

---

## 4. Index Documents

### Create Indexing Job

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "/data/documents",
    "index_name": "company_docs_v1"
  }'
```

Response:

```json
{"job_id": "550e8400-e29b-41d4-a716-446655440000", "message": "Indexing job created successfully"}
```

### Monitor Progress

```bash
curl http://localhost:8000/index/{job_id}
```

Response fields:

| Field | Description |
|-------|-------------|
| `status` | pending, running, completed, failed, cancelled |
| `current_stage` | queued, ingest, chunk, embed, index, finalize |
| `progress_percent` | 0.0 - 100.0 |

### List Completed Indices

```bash
curl http://localhost:8000/indices
```

---

## 5. Query Documents

### Basic Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the company policy on remote work?",
    "index_name": "company_docs_v1",
    "top_k": 5,
    "clearance_level": 1
  }'
```

### Clearance Levels

| Level | Name | Access |
|-------|------|--------|
| 0 | PUBLIC | Public documents only |
| 1 | INTERNAL | Public + Internal |
| 2 | CONFIDENTIAL | Public + Internal + Confidential |
| 3 | SECRET | All except Top Secret |
| 4 | TOP_SECRET | All documents |

### Response Interpretation

**Successful query:**

```json
{
  "answer": "The company allows remote work up to 3 days per week...",
  "sources": [...],
  "query": "...",
  "refused": false,
  "refusal_reason": null
}
```

**Refused query:**

```json
{
  "answer": "I cannot provide an answer due to insufficient evidence...",
  "sources": [],
  "query": "...",
  "refused": true,
  "refusal_reason": "insufficient_evidence"
}
```

---

## 6. Stateless Horizontal Scaling

### Architecture

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └────────┬────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  Worker 1   │   │  Worker 2   │   │  Worker N   │
    │  (uvicorn)  │   │  (uvicorn)  │   │  (uvicorn)  │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             ▼
                    ┌─────────────────┐
                    │     MySQL       │
                    │  (shared state) │
                    └─────────────────┘
```

### Demonstration

**Terminal 1 - Start multiple workers:**

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Terminal 2 - Create indexing job:**

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"source_path": "/data/docs", "index_name": "test_index"}'
```

**Terminal 3 - Poll status from different process:**

```bash
while true; do
  curl -s http://localhost:8000/index/{job_id} | jq '.progress_percent, .status'
  sleep 1
done
```

Any worker can serve the status request because all state is in MySQL.

### Verify Statelessness

1. Start job on worker A
2. Kill worker A mid-progress
3. Query status from worker B
4. Status reflects progress (state persisted in MySQL)

```bash
# Check which worker handles requests
curl -v http://localhost:8000/health 2>&1 | grep "< "
```

---

## Troubleshooting

| Symptom | Cause | Resolution |
|---------|-------|------------|
| `database: unhealthy` | MySQL connection failed | Verify MySQL is running, check credentials in `src/core/config.py` |
| Job stuck at `pending` | Background task not started | Check uvicorn logs for errors |
| `refused: true` | Insufficient evidence or clearance | Verify index exists, check clearance level |
| Slow embeddings | Ollama overloaded | Increase Ollama resources or reduce batch size |

---

## Monitoring Queries

```sql
-- Query audit logs
SELECT
  DATE(created_at) as date,
  COUNT(*) as total_queries,
  SUM(refused) as refused_count,
  AVG(latency_ms) as avg_latency_ms
FROM query_audit_logs
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Top refusal reasons
SELECT
  refusal_reason,
  COUNT(*) as count
FROM query_audit_logs
WHERE refused = 1
GROUP BY refusal_reason;
```
