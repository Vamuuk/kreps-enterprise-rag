"""
Document ingestion module for enterprise RAG system.
Supports PDF, TXT, and MD files with metadata preservation.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import fitz  # PyMuPDF

from config import RAW_DOCS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents from filesystem with metadata extraction."""

    def __init__(self, docs_dir: Path = RAW_DOCS_DIR):
        self.docs_dir = docs_dir

    def load_all(self) -> List[Dict]:
        """
        Recursively load all supported documents from raw_docs directory.

        Returns:
            List of document dicts with text and metadata
        """
        documents = []

        if not self.docs_dir.exists():
            logger.warning(f"Documents directory does not exist: {self.docs_dir}")
            return documents

        # Supported file extensions
        patterns = ["*.pdf", "*.txt", "*.md"]

        for pattern in patterns:
            for file_path in self.docs_dir.rglob(pattern):
                logger.info(f"Loading: {file_path.name}")
                try:
                    if pattern == "*.pdf":
                        docs = self._load_pdf(file_path)
                    else:
                        docs = self._load_text(file_path)

                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} pages/sections from {file_path.name}")

                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    continue

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def _load_pdf(self, file_path: Path) -> List[Dict]:
        """
        Load PDF file page by page.

        Args:
            file_path: Path to PDF file

        Returns:
            List of page documents with metadata
        """
        documents = []

        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text()

                if text.strip():  # Skip empty pages
                    documents.append({
                        "text": text,
                        "metadata": {
                            "document": file_path.name,
                            "page": page_num + 1,  # 1-indexed for user display
                            "path": str(file_path)
                        }
                    })

        return documents

    def _load_text(self, file_path: Path) -> List[Dict]:
        """
        Load TXT or MD file.

        Args:
            file_path: Path to text file

        Returns:
            Single document dict (no page numbers for text files)
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if not text.strip():
            return []

        return [{
            "text": text,
            "metadata": {
                "document": file_path.name,
                "page": None,
                "path": str(file_path)
            }
        }]


def ingest_documents() -> List[Dict]:
    """
    Main ingestion entry point.

    Returns:
        List of loaded documents with metadata
    """
    loader = DocumentLoader()
    return loader.load_all()


if __name__ == "__main__":
    # Test ingestion
    docs = ingest_documents()
    print(f"Ingested {len(docs)} document pages/sections")
