"""
Configuration management for enterprise RAG system.
All paths and parameters are centralized here.
"""

import os
from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
PROCESSED_DIR = DATA_DIR / "processed"

# Storage directories
STORAGE_DIR = PROJECT_ROOT / "storage"
FAISS_DIR = STORAGE_DIR / "faiss"
BM25_DIR = STORAGE_DIR / "bm25"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Processed data files
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

# Chunking parameters
CHUNK_SIZE_TOKENS = 700  # Target 600-800
CHUNK_OVERLAP_TOKENS = 100

# Retrieval parameters
TOP_K_FAISS = 10
TOP_K_BM25 = 10
SEMANTIC_WEIGHT = 0.7
LEXICAL_WEIGHT = 0.3

# Answer guardrails
MIN_SOURCES = 2
CONFIDENCE_THRESHOLD_HIGH = 0.75
CONFIDENCE_THRESHOLD_MEDIUM = 0.50

# Embedding dimension (placeholder - will match Qwen model)
EMBEDDING_DIM = 768  # Standard dimension, adjust when Qwen is integrated


def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        RAW_DOCS_DIR,
        PROCESSED_DIR,
        FAISS_DIR,
        BM25_DIR,
        LOGS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
