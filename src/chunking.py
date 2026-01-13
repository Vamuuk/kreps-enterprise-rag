"""
Enterprise-grade structured chunking module.
Implements:
- Section-aware, token-aware chunking with overlap
- Adaptive chunk sizes based on document type
- Enhanced heading preservation in metadata
"""

import json
import logging
import hashlib
import re
from typing import List, Dict

import tiktoken

from src.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNKS_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """Token-aware text chunker with adaptive sizing and section detection."""

    # Adaptive chunk sizes based on document type
    CHUNK_SIZES = {
        "policy": 500,      # Smaller chunks for policies (precise retrieval)
        "report": 900,      # Larger chunks for reports (context-heavy)
        "manual": 700,      # Medium chunks for manuals
        "other": 700        # Default size
    }

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        overlap: int = CHUNK_OVERLAP_TOKENS
    ):
        self.default_chunk_size = chunk_size
        self.overlap = overlap
        # Use cl100k_base encoding (GPT-4 tokenizer, widely compatible)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk all documents with metadata preservation and adaptive sizing.

        Args:
            documents: List of document dicts from ingest module

        Returns:
            List of chunk dicts with chunk_id and enhanced metadata
        """
        all_chunks = []

        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]

            # Determine chunk size based on document type
            doc_type = metadata.get("document_type", "other")
            chunk_size = self.CHUNK_SIZES.get(doc_type, self.default_chunk_size)

            logger.debug(
                f"Chunking {metadata['document']} (type={doc_type}, chunk_size={chunk_size})"
            )

            # Try section-based splitting first
            sections = self._detect_sections(text)

            if len(sections) > 1:
                # Process each section separately
                for section_title, section_text in sections:
                    chunks = self._chunk_text(section_text, chunk_size)
                    for chunk_idx, chunk_text in enumerate(chunks):
                        chunk = self._create_chunk(
                            text=chunk_text,
                            metadata=metadata,
                            section=section_title,
                            chunk_index=chunk_idx
                        )
                        all_chunks.append(chunk)
            else:
                # No clear sections, chunk entire text
                chunks = self._chunk_text(text, chunk_size)
                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk = self._create_chunk(
                        text=chunk_text,
                        metadata=metadata,
                        section="Main Content",
                        chunk_index=chunk_idx
                    )
                    all_chunks.append(chunk)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def _detect_sections(self, text: str) -> List[tuple[str, str]]:
        """
        Detect sections in text based on headers with enhanced pattern matching.

        Args:
            text: Document text

        Returns:
            List of (section_title, section_text) tuples
        """
        # Enhanced header patterns for better heading detection
        # Examples: "1. Introduction", "Section 2:", "# Header", "INTRODUCTION", etc.
        patterns = [
            r'^(?:#+\s+)(.+)$',  # Markdown headers: # Title
            r'^(?:\d+\.?\s+)(.+)$',  # Numbered: 1. Title or 1 Title
            r'^([A-Z][A-Z\s]{2,}):?\s*$',  # ALL CAPS HEADERS
            r'^(?:Section|Chapter)\s+\d+[:\s]+(.+)$',  # Section 1: Title
        ]

        lines = text.split('\n')
        sections = []
        current_section = "Introduction"
        current_text = []

        for line in lines:
            is_header = False

            # Check all patterns
            for pattern in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # Save previous section
                    if current_text:
                        sections.append((current_section, '\n'.join(current_text)))

                    # Start new section
                    current_section = match.group(1).strip()
                    current_text = []
                    is_header = True
                    break

            if not is_header:
                current_text.append(line)

        # Add final section
        if current_text:
            sections.append((current_section, '\n'.join(current_text)))

        # If only one section found, return empty (no clear sections)
        if len(sections) <= 1:
            return []

        return sections

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into token-aware chunks with overlap.

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in tokens

        Returns:
            List of text chunks
        """
        # Tokenize text
        tokens = self.tokenizer.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            # Get chunk of tokens
            end = start + chunk_size
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Clean up chunk
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Move start position with overlap
            start += chunk_size - self.overlap

        return chunks

    def _create_chunk(
        self,
        text: str,
        metadata: Dict,
        section: str,
        chunk_index: int
    ) -> Dict:
        """
        Create chunk dict with deterministic ID and enhanced metadata.

        Args:
            text: Chunk text
            metadata: Source document metadata
            section: Section name (heading)
            chunk_index: Index within section

        Returns:
            Chunk dict with chunk_id, text, and enhanced metadata
        """
        # Extract heading from text (first line if it looks like a heading)
        lines = text.split('\n', 2)
        heading = None
        if lines and len(lines[0]) < 100:  # Likely a heading
            first_line = lines[0].strip()
            if re.match(r'^(?:#+\s+|\d+\.?\s+|[A-Z][A-Z\s]{2,})', first_line):
                heading = first_line

        # Create deterministic chunk ID
        id_string = (
            f"{metadata['document']}|{metadata.get('page')}|"
            f"{section}|{chunk_index}|{text[:100]}"
        )
        chunk_id = hashlib.sha256(id_string.encode()).hexdigest()[:16]

        return {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": {
                "document": metadata["document"],
                "page": metadata.get("page"),
                "section": section,
                "heading": heading,  # Preserved heading
                "language": metadata.get("language", "unknown"),
                "document_type": metadata.get("document_type", "other"),
                "year": metadata.get("year"),
                "chunk_index": chunk_index
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
    from src.ingest import ingest_documents

    docs = ingest_documents()
    chunks = chunk_and_save(docs)
    print(f"Created {len(chunks)} chunks")

    # Show sample chunk metadata
    if chunks:
        sample = chunks[0]
        print(f"\nSample chunk metadata:")
        print(f"  Document: {sample['metadata']['document']}")
        print(f"  Section: {sample['metadata']['section']}")
        print(f"  Heading: {sample['metadata']['heading']}")
        print(f"  Language: {sample['metadata']['language']}")
        print(f"  Type: {sample['metadata']['document_type']}")
