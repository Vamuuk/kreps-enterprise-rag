"""
Main CLI application for KREPS enterprise RAG system.

Usage:
    python src/app.py index    # Build indices from raw documents
    python src/app.py query "your question"  # Query the system
"""

import sys
import logging
from pathlib import Path

import click

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ensure_directories
from src.ingest import ingest_documents
from src.chunking import chunk_and_save, load_chunks
from src.index_faiss import build_faiss_index
from src.index_bm25 import build_bm25_index
from src.answer import answer_query

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """KREPS Enterprise RAG System - Offline Knowledge Assistant"""
    pass


@cli.command()
def index():
    """
    Build indices from documents in data/raw_docs/.

    Steps:
    1. Ingest documents (PDF, TXT, MD)
    2. Chunk documents with overlap
    3. Build FAISS semantic index (requires Qwen embeddings)
    4. Build BM25 lexical index
    """
    click.echo("=" * 60)
    click.echo("KREPS RAG SYSTEM - INDEXING")
    click.echo("=" * 60)
    click.echo()

    # Ensure directories exist
    ensure_directories()

    # Step 1: Ingest documents
    click.echo("[1/4] Ingesting documents...")
    documents = ingest_documents()

    if not documents:
        click.echo("ERROR: No documents found in data/raw_docs/", err=True)
        click.echo("Please add PDF, TXT, or MD files to data/raw_docs/", err=True)
        sys.exit(1)

    click.echo(f"✓ Ingested {len(documents)} document pages/sections")
    click.echo()

    # Step 2: Chunk documents
    click.echo("[2/4] Chunking documents...")
    chunks = chunk_and_save(documents)
    click.echo(f"✓ Created {len(chunks)} chunks")
    click.echo()

    # Step 3: Build FAISS index
    click.echo("[3/4] Building FAISS semantic index...")
    click.echo("NOTE: This requires Qwen embedding model integration")

    try:
        build_faiss_index(chunks)
        click.echo("✓ FAISS index built successfully")
    except NotImplementedError:
        click.echo("⚠ FAISS indexing skipped (Qwen embeddings not integrated)", fg="yellow")
        click.echo("  The system will use BM25 lexical search only")
        click.echo("  To enable semantic search, integrate Qwen in src/embed.py")

    click.echo()

    # Step 4: Build BM25 index
    click.echo("[4/4] Building BM25 lexical index...")
    build_bm25_index(chunks)
    click.echo("✓ BM25 index built successfully")
    click.echo()

    click.echo("=" * 60)
    click.echo("INDEXING COMPLETE")
    click.echo("=" * 60)
    click.echo()
    click.echo("You can now query the system:")
    click.echo('  python src/app.py query "your question"')
    click.echo()


@cli.command()
@click.argument('question')
def query(question):
    """
    Query the RAG system.

    QUESTION: Your query as a string
    """
    click.echo("=" * 60)
    click.echo("KREPS RAG SYSTEM - QUERY")
    click.echo("=" * 60)
    click.echo()
    click.echo(f"Query: {question}")
    click.echo()

    # Execute query
    try:
        result = answer_query(question)
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        click.echo("Have you run indexing? Try: python src/app.py index", err=True)
        sys.exit(1)

    # Display results
    click.echo("-" * 60)
    click.echo("ANSWER")
    click.echo("-" * 60)
    click.echo(result["answer"])
    click.echo()

    click.echo("-" * 60)
    click.echo(f"CONFIDENCE: {result['confidence']}")
    click.echo("-" * 60)
    click.echo()

    # Display sources
    if result["sources"]:
        click.echo("-" * 60)
        click.echo(f"SOURCES ({len(result['sources'])} documents)")
        click.echo("-" * 60)
        for i, source in enumerate(result["sources"], 1):
            page_info = f"Page {source['page']}" if source['page'] else "N/A"
            click.echo(f"{i}. {source['document']} - {page_info}")
            click.echo(f"   Section: {source['section']}")
            click.echo(f"   Relevance: {source['score']:.3f}")
            click.echo()

    # Display chunk count
    if result["chunks"]:
        click.echo(f"Retrieved {len(result['chunks'])} relevant chunks")
        click.echo()

    click.echo("=" * 60)


@cli.command()
def status():
    """Check system status and index information."""
    click.echo("=" * 60)
    click.echo("KREPS RAG SYSTEM - STATUS")
    click.echo("=" * 60)
    click.echo()

    from src.config import CHUNKS_FILE, FAISS_DIR, BM25_DIR

    # Check chunks
    if CHUNKS_FILE.exists():
        chunks = load_chunks()
        click.echo(f"✓ Chunks: {len(chunks)} indexed")
    else:
        click.echo("✗ Chunks: Not found")

    # Check FAISS index
    faiss_index = FAISS_DIR / "index.faiss"
    if faiss_index.exists():
        click.echo(f"✓ FAISS index: Available")
    else:
        click.echo("✗ FAISS index: Not found")

    # Check BM25 index
    bm25_index = BM25_DIR / "bm25.pkl"
    if bm25_index.exists():
        click.echo(f"✓ BM25 index: Available")
    else:
        click.echo("✗ BM25 index: Not found")

    click.echo()

    if not (CHUNKS_FILE.exists() and bm25_index.exists()):
        click.echo("Run indexing: python src/app.py index")

    click.echo()


if __name__ == "__main__":
    cli()
