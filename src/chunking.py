"""
Enterprise-grade structured chunking module.
Implements section-aware, token-aware chunking with overlap.
"""

import json
import logging
import hashlib
import re
from typing import List, Dict

import tiktoken

from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNKS_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """Token-aware text chunker with section detection."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        overlap: int = CHUNK_OVERLAP_TOKENS
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Use cl100k_base encoding (GPT-4 tokenizer, widely compatible)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk all documents with metadata preservation.

        Args:
            documents: List of document dicts from ingest module

        Returns:
            List of chunk dicts with chunk_id and metadata
        """
        all_chunks = []

        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]

            # Try section-based splitting first
            sections = self._detect_sections(text)

            if len(sections) > 1:
                # Process each section separately
                for section_title, section_text in sections:
                    chunks = self._chunk_text(section_text)
                    for chunk_text in chunks:
                        chunk = self._create_chunk(
                            text=chunk_text,
                            document=metadata["document"],
                            page=metadata["page"],
                            section=section_title
                        )
                        all_chunks.append(chunk)
            else:
                # No clear sections, chunk entire text
                chunks = self._chunk_text(text)
                for chunk_text in chunks:
                    chunk = self._create_chunk(
                        text=chunk_text,
                        document=metadata["document"],
                        page=metadata["page"],
                        section="Main Content"
                    )
                    all_chunks.append(chunk)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def _detect_sections(self, text: str) -> List[tuple[str, str]]:
        """
        Detect sections in text based on headers.

        Args:
            text: Document text

        Returns:
            List of (section_title, section_text) tuples
        """
        # Match common header patterns
        # Examples: "1. Introduction", "Section 2:", "# Header", etc.
        pattern = r'^(?:#+\s+|\d+\.?\s+|[A-Z][A-Z\s]{2,}:)\s*(.+)$'

        lines = text.split('\n')
        sections = []
        current_section = "Introduction"
        current_text = []

        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                # Save previous section
                if current_text:
                    sections.append((current_section, '\n'.join(current_text)))

                # Start new section
                current_section = match.group(1).strip()
                current_text = []
            else:
                current_text.append(line)

        # Add final section
        if current_text:
            sections.append((current_section, '\n'.join(current_text)))

        # If only one section found, return empty (no clear sections)
        if len(sections) <= 1:
            return []

        return sections

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into token-aware chunks with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        # Tokenize text
        tokens = self.tokenizer.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            # Get chunk of tokens
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Clean up chunk
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Move start position with overlap
            start += self.chunk_size - self.overlap

        return chunks

    def _create_chunk(
        self,
        text: str,
        document: str,
        page: int | None,
        section: str
    ) -> Dict:
        """
        Create chunk dict with deterministic ID.

        Args:
            text: Chunk text
            document: Source document name
            page: Page number (or None)
            section: Section name

        Returns:
            Chunk dict with chunk_id, text, and metadata
        """
        # Create deterministic chunk ID
        id_string = f"{document}|{page}|{section}|{text[:100]}"
        chunk_id = hashlib.sha256(id_string.encode()).hexdigest()[:16]

        return {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": {
                "document": document,
                "page": page,
                "section": section
            }
        }


def chunk_and_save(documents: List[Dict]) -> List[Dict]:
    """
    Chunk documents and save to JSONL file.

    Args:
        documents: List of ingested documents

    Returns:
        List of chunks
    """
    chunker = TextChunker()
    chunks = chunker.chunk_documents(documents)

    # Save to JSONL
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(chunks)} chunks to {CHUNKS_FILE}")
    return chunks


def load_chunks() -> List[Dict]:
    """
    Load chunks from JSONL file.

    Returns:
        List of chunk dicts
    """
    if not CHUNKS_FILE.exists():
        logger.warning(f"Chunks file not found: {CHUNKS_FILE}")
        return []

    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
    return chunks


if __name__ == "__main__":
    # Test chunking
    from ingest import ingest_documents

    docs = ingest_documents()
    chunks = chunk_and_save(docs)
    print(f"Created {len(chunks)} chunks")
