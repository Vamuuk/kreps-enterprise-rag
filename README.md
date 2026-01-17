# KREPS RAG System

Offline document search system using hybrid retrieval (FAISS + BM25) and answer generation via Qwen.

## What it does

The system works like this:
1. Loads PDF/TXT/MD files
2. Splits them into chunks
3. Indexes via FAISS (semantic) and BM25 (keywords)
4. Searches for similar chunks on query
5. Generates answer via Qwen LLM

## Project structure

```
kREPS-rag/
├── src/                  # Code
│   ├── app.py           # CLI interface
│   ├── ingest.py        # Document loading
│   ├── chunking.py      # Splitting into chunks
│   ├── embed.py         # Embeddings via Ollama
│   ├── index_faiss.py   # Semantic search
│   ├── index_bm25.py    # Lexical search
│   ├── retrieve.py      # Hybrid search
│   └── answer.py        # Answer generation
├── data/
│   └── raw_docs/        # Put documents here
├── storage/             # Indexes stored here
└── frontend.py          # Web interface (Streamlit)
```

## Installation

1. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Check Ollama is running:
```bash
ollama list
# Should have: nomic-embed-text and qwen2.5:3b
```

If models missing:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

## How to use

### 1. Add documents
Put PDF/TXT/MD files in `data/raw_docs/`

### 2. Create index
```bash
python src/app.py index
```

This will:
- Load documents
- Split into chunks
- Detect language and document type
- Create FAISS and BM25 indexes

### 3. Ask questions

Via CLI:
```bash
python src/app.py query "What are the safety procedures?"
```

Via web interface:
```bash
python -m streamlit run frontend.py
```

### 4. Check status
```bash
python src/app.py status
```

## Features

**Auto metadata detection:**
- Document language (en/ru)
- Type (policy/report/manual)
- Publication year

**Adaptive chunking:**
- Policies → 500 tokens (precise search)
- Reports → 900 tokens (more context)
- Manuals → 700 tokens

**Language prioritization:**
- Russian query → priority to Russian docs
- English query → priority to English docs

**Quality guardrails:**
- Minimum sources (MIN_SOURCES)
- Relevance threshold
- Refuse on weak evidence
- Answer ONLY from context

## Settings

In `src/config.py`:
```python
CHUNK_SIZE_TOKENS = 700          # Chunk size
CHUNK_OVERLAP_TOKENS = 100       # Overlap
SEMANTIC_WEIGHT = 0.7            # Semantic weight
LEXICAL_WEIGHT = 0.3             # BM25 weight
MIN_SOURCES = 2                  # Min sources
CONFIDENCE_THRESHOLD_HIGH = 0.75 # Confidence threshold
```

## How search works

1. **Query language detection** (cyrillic/latin)
2. **FAISS search** → top-10 semantically similar
3. **BM25 search** → top-10 by keywords
4. **Merge + normalize** → hybrid score
5. **Language boost** → +15% for matching language
6. **Top-5 chunks** → into Qwen prompt

## Examples

```bash
# Indexing
python src/app.py index

# Query
python src/app.py query "What is safety?"

# Status
python src/app.py status

# Web interface
python -m streamlit run frontend.py
```

## Common issues

**ModuleNotFoundError: No module named 'faiss'**
```bash
pip install faiss-cpu
```

**ModuleNotFoundError: No module named 'rank_bm25'**
```bash
pip install rank-bm25
```

**Connection refused (Ollama)**
- Check Ollama is running: `ollama list`
- Check port: `http://localhost:11434`

**Generation timeout**
- Increase timeout in `src/answer.py` (currently 450 sec)
- Or use lighter model

## Query result

```python
{
    "answer": "Answer from documents",
    "confidence": "High",  # High/Medium/Low
    "sources": [
        {
            "document": "safety_policy.pdf",
            "page": 5,
            "section": "Procedures",
            "score": 0.87
        }
    ],
    "chunks": [...]  # Retrieved chunks
}
```

## Dependencies

- Python 3.9+
- Ollama (for embeddings and LLM)
- FAISS (vector search)
- BM25 (lexical search)
- Streamlit (web interface)

## License

Student project
