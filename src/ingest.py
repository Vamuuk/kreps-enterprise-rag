"""
Document ingestion module for enterprise RAG system.
Supports PDF, TXT, and MD files with enhanced metadata extraction:
- Language detection
- Document type classification
- Version/year extraction
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

import fitz  # PyMuPDF

from src.config import RAW_DOCS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract enhanced metadata from documents."""

    # Document type keywords for classification
    TYPE_KEYWORDS = {
        "policy": ["policy", "policies", "regulation", "compliance", "guideline", "standard"],
        "report": ["report", "analysis", "findings", "summary", "assessment", "evaluation"],
        "manual": ["manual", "guide", "handbook", "instruction", "tutorial", "documentation"],
    }

    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect language from text using simple heuristics.

        Args:
            text: Document text

        Returns:
            Language code (en, ru, etc.) or 'unknown'
        """
        # Simple offline language detection based on character patterns
        # For production, you could add langdetect library, but keeping minimal for now

        # Check for Cyrillic characters (Russian)
        cyrillic_count = len(re.findall(r'[а-яА-ЯёЁ]', text[:1000]))
        # Check for Latin characters (English)
        latin_count = len(re.findall(r'[a-zA-Z]', text[:1000]))

        if cyrillic_count > latin_count * 0.3:
            return "ru"
        elif latin_count > 50:
            return "en"
        else:
            return "unknown"

    @staticmethod
    def classify_document_type(filename: str, text: str) -> str:
        """
        Classify document type based on filename and content.

        Args:
            filename: Document filename
            text: Document text (first few pages)

        Returns:
            Document type: policy, report, manual, or other
        """
        # Combine filename and first 2000 chars for classification
        search_text = (filename + " " + text[:2000]).lower()

        for doc_type, keywords in MetadataExtractor.TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in search_text:
                    return doc_type

        return "other"

    @staticmethod
    def extract_year(text: str) -> Optional[int]:
        """
        Extract publication year from document text.

        Args:
            text: Document text

        Returns:
            Year as integer, or None if not found
        """
        # Look for year patterns in first 3000 chars (title page area)
        header_text = text[:3000]

        # Pattern 1: "2023", "2024" as standalone year
        # Pattern 2: "Published: 2023", "Date: 2024"
        # Pattern 3: "Copyright 2023", "(c) 2024"
        patterns = [
            r'\b(19\d{2}|20[0-2]\d)\b',  # Years 1900-2029
            r'(?:published|date|year|copyright|\(c\))[\s:]*(\d{4})',
        ]

        years_found = []
        for pattern in patterns:
            matches = re.findall(pattern, header_text, re.IGNORECASE)
            years_found.extend([int(m) if isinstance(m, str) else int(m) for m in matches])

        # Return most recent valid year (2000-2030)
        valid_years = [y for y in years_found if 2000 <= y <= 2030]
        if valid_years:
            return max(valid_years)

        return None


class DocumentLoader:
    """Loads documents from filesystem with enhanced metadata extraction."""

    def __init__(self, docs_dir: Path = RAW_DOCS_DIR):
        self.docs_dir = docs_dir
        self.metadata_extractor = MetadataExtractor()

    def load_all(self) -> List[Dict]:
        """
        Recursively load all supported documents from raw_docs directory.

        Returns:
            List of document dicts with text and enhanced metadata
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
        Load PDF file page by page with enhanced metadata.

        Args:
            file_path: Path to PDF file

        Returns:
            List of page documents with metadata
        """
        documents = []
        full_text = ""  # Accumulate text for document-level metadata

        with fitz.open(file_path) as pdf:
            # Extract full text for metadata
            for page_num in range(min(5, len(pdf))):  # First 5 pages for metadata
                full_text += pdf[page_num].get_text()

            # Extract document-level metadata
            language = self.metadata_extractor.detect_language(full_text)
            doc_type = self.metadata_extractor.classify_document_type(
                file_path.name, full_text
            )
            year = self.metadata_extractor.extract_year(full_text)

            # Load pages
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text()

                if text.strip():  # Skip empty pages
                    documents.append({
                        "text": text,
                        "metadata": {
                            "document": file_path.name,
                            "page": page_num + 1,  # 1-indexed
                            "path": str(file_path),
                            "language": language,
                            "document_type": doc_type,
                            "year": year
                        }
                    })

        logger.info(
            f"  Metadata: language={language}, type={doc_type}, year={year}"
        )

        return documents

    def _load_text(self, file_path: Path) -> List[Dict]:
        """
        Load TXT or MD file with enhanced metadata.

        Args:
            file_path: Path to text file

        Returns:
            Single document dict with metadata
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if not text.strip():
            return []

        # Extract metadata
        language = self.metadata_extractor.detect_language(text)
        doc_type = self.metadata_extractor.classify_document_type(file_path.name, text)
        year = self.metadata_extractor.extract_year(text)

        logger.info(
            f"  Metadata: language={language}, type={doc_type}, year={year}"
        )

        return [{
            "text": text,
            "metadata": {
                "document": file_path.name,
                "page": None,
                "path": str(file_path),
                "language": language,
                "document_type": doc_type,
                "year": year
            }
        }]


def ingest_documents() -> List[Dict]:
    """
    Main ingestion entry point.

    Returns:
        List of loaded documents with enhanced metadata
    """
    loader = DocumentLoader()
    return loader.load_all()


if __name__ == "__main__":
    # Test ingestion
    docs = ingest_documents()
    print(f"Ingested {len(docs)} document pages/sections")

    # Show sample metadata
    if docs:
        sample = docs[0]["metadata"]
        print(f"\nSample metadata:")
        print(f"  Document: {sample['document']}")
        print(f"  Language: {sample['language']}")
        print(f"  Type: {sample['document_type']}")
        print(f"  Year: {sample['year']}")
