# KREPS Enterprise RAG System

Offline enterprise knowledge system for technical documentation in safety-critical industrial environments.

## Status

**Foundation Complete** ✓
**Qwen LLM Integration** ⏳ Pending

This system implements all RAG components EXCEPT LLM inference. The retrieval pipeline, indexing, and guardrails are production-ready. Qwen embedding and generation models must be integrated before deployment.

## Architecture

```
Document Ingestion → Chunking → Indexing → Retrieval → Answer Generation
                                  ↓
                            FAISS (semantic) + BM25 (lexical)
                                  ↓
                            Hybrid Retrieval
                                  ↓
                            Guardrails + LLM
```

## Project Structure

```
kREPS-rag/
├── src/
│   ├── app.py           # CLI interface
│   ├── contracts.py     # Interface definitions
│   ├── config.py        # Configuration
│   ├── ingest.py        # Document loading (PDF/TXT/MD)
│   ├── chunking.py      # Token-aware chunking
│   ├── embed.py         # Embedding engine (PLACEHOLDER)
│   ├── index_faiss.py   # Semantic indexing
│   ├── index_bm25.py    # Lexical indexing
│   ├── retrieve.py      # Hybrid retrieval
│   └── answer.py        # Answer generation (PLACEHOLDER)
├── data/
│   ├── raw_docs/        # Place your documents here
│   └── processed/       # Generated chunks
├── storage/
│   ├── faiss/           # Vector index
│   └── bm25/            # Lexical index
├── logs/
├── requirements.txt
└── README.md
```

## Setup

### 1. Create Virtual Environment

```bash
cd kREPS-rag
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Documents

Place your PDF, TXT, or MD files in `data/raw_docs/`:

```bash
# Windows
copy your_docs/*.pdf data\raw_docs\

# Linux/Mac
cp your_docs/*.pdf data/raw_docs/
```

## Usage

### Build Indices

```bash
python src/app.py index
```

This will:
1. Ingest documents from `data/raw_docs/`
2. Chunk text with 700-token windows and 100-token overlap
3. Build FAISS semantic index (requires Qwen embeddings)
4. Build BM25 lexical index

**Note:** FAISS indexing will be skipped until Qwen embeddings are integrated. The system will use BM25-only retrieval.

### Query System

```bash
python src/app.py query "What are the safety procedures?"
```

### Check Status

```bash
python src/app.py status
```

## Current Functionality

### ✓ Working Now

- Document ingestion (PDF page-by-page, TXT, MD)
- Section-aware chunking with token overlap
- BM25 lexical search (no external dependencies)
- Hybrid retrieval framework
- Query guardrails (minimum sources, confidence scoring)
- Structured output (answer, confidence, sources, chunks)

### ⏳ Requires Integration

- Qwen embedding model (for semantic search)
- Qwen generation model (for answer synthesis)

## Integration Points

### 1. Qwen Embedding Model

**File:** `src/embed.py`
**Function:** `EmbeddingEngine.embed_texts()`

Replace NotImplementedError with:

```python
def embed_texts(self, texts: List[str]) -> List[List[float]]:
    # Load Qwen embedding model (offline)
    # Tokenize and batch process
    # Return list of vectors
    pass
```

### 2. Qwen Generation Model

**File:** `src/answer.py`
**Function:** `AnswerGenerator._generate_placeholder_answer()`

Replace placeholder logic with:

```python
def _generate_answer(self, query: str, chunks: List[Dict]) -> str:
    # Construct prompt with query and retrieved context
    # Call Qwen LLM for inference
    # Extract and return generated answer
    pass
```

## Configuration

Edit `src/config.py` to adjust:

- `CHUNK_SIZE_TOKENS`: 700 (target 600-800)
- `CHUNK_OVERLAP_TOKENS`: 100
- `SEMANTIC_WEIGHT`: 0.7 (hybrid retrieval)
- `LEXICAL_WEIGHT`: 0.3
- `CONFIDENCE_THRESHOLD_HIGH`: 0.75
- `CONFIDENCE_THRESHOLD_MEDIUM`: 0.50

## Output Contract

All queries return:

```python
{
    "answer": str,
    "confidence": "High" | "Medium" | "Low",
    "sources": [
        {
            "document": str,
            "page": int,
            "section": str,
            "score": float
        }
    ],
    "chunks": [
        {
            "chunk_id": str,
            "text": str,
            "metadata": dict,
            "score": float
        }
    ]
}
```

## Testing Without Qwen

The system can be tested with BM25-only retrieval:

1. Add documents to `data/raw_docs/`
2. Run `python src/app.py index`
3. FAISS indexing will be skipped (expected)
4. Query with `python src/app.py query "question"`
5. Results will show BM25 retrieval working
6. Answer will be placeholder (no LLM)

## Enterprise Features

- **Offline-first:** No external API calls
- **Deterministic chunk IDs:** Stable across re-indexing
- **Section-aware chunking:** Preserves document structure
- **Hybrid retrieval:** Combines semantic and lexical search
- **Quality guardrails:** Minimum sources, score thresholds
- **Confidence scoring:** High/Medium/Low based on retrieval
- **Structured logging:** Enterprise observability

## Next Steps

1. Integrate Qwen embedding model in `src/embed.py`
2. Test FAISS indexing with embeddings
3. Integrate Qwen generation model in `src/answer.py`
4. Test end-to-end query pipeline
5. Tune chunking and retrieval parameters
6. Deploy to production environment

## License

Internal enterprise use only.
