# Загрузка документов из папки raw_docs
# Поддерживает PDF, TXT, MD
# Определяет язык, тип документа и год

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

import fitz  # PyMuPDF

from src.config import RAW_DOCS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataExtractor:
    # Ключевые слова для определения типа документа
    TYPE_KEYWORDS = {
        "policy": ["policy", "policies", "regulation", "compliance", "guideline", "standard"],
        "report": ["report", "analysis", "findings", "summary", "assessment", "evaluation"],
        "manual": ["manual", "guide", "handbook", "instruction", "tutorial", "documentation"],
    }

    @staticmethod
    def detect_language(text: str) -> str:
        # Простое определение языка по кириллице/латинице
        cyrillic_count = len(re.findall(r'[а-яА-ЯёЁ]', text[:1000]))
        latin_count = len(re.findall(r'[a-zA-Z]', text[:1000]))

        if cyrillic_count > latin_count * 0.3:
            return "ru"
        elif latin_count > 50:
            return "en"
        else:
            return "unknown"

    @staticmethod
    def classify_document_type(filename: str, text: str) -> str:
        # Определяем тип документа по названию и содержимому
        search_text = (filename + " " + text[:2000]).lower()

        for doc_type, keywords in MetadataExtractor.TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in search_text:
                    return doc_type

        return "other"

    @staticmethod
    def extract_year(text: str) -> Optional[int]:
        # Ищем год публикации в первых 3000 символах
        header_text = text[:3000]

        patterns = [
            r'\b(19\d{2}|20[0-2]\d)\b',  # 1900-2029
            r'(?:published|date|year|copyright|\(c\))[\s:]*(\d{4})',
        ]

        years_found = []
        for pattern in patterns:
            matches = re.findall(pattern, header_text, re.IGNORECASE)
            years_found.extend([int(m) if isinstance(m, str) else int(m) for m in matches])

        # Берем самый свежий год из 2000-2030
        valid_years = [y for y in years_found if 2000 <= y <= 2030]
        if valid_years:
            return max(valid_years)

        return None


class DocumentLoader:
    def __init__(self, docs_dir: Path = RAW_DOCS_DIR):
        self.docs_dir = docs_dir
        self.metadata_extractor = MetadataExtractor()

    def load_all(self) -> List[Dict]:
        # Загружаем все PDF/TXT/MD из raw_docs
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
        # Загружаем PDF постранично + метаданные
        documents = []
        full_text = ""

        with fitz.open(file_path) as pdf:
            # Берем первые 5 страниц для определения метаданных
            for page_num in range(min(5, len(pdf))):
                full_text += pdf[page_num].get_text()

            language = self.metadata_extractor.detect_language(full_text)
            doc_type = self.metadata_extractor.classify_document_type(file_path.name, full_text)
            year = self.metadata_extractor.extract_year(full_text)

            # Загружаем все страницы
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text()

                if text.strip():
                    documents.append({
                        "text": text,
                        "metadata": {
                            "document": file_path.name,
                            "page": page_num + 1,
                            "path": str(file_path),
                            "language": language,
                            "document_type": doc_type,
                            "year": year
                        }
                    })

        logger.info(f"  Metadata: language={language}, type={doc_type}, year={year}")

        return documents

    def _load_text(self, file_path: Path) -> List[Dict]:
        # Загружаем TXT/MD файл
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if not text.strip():
            return []

        language = self.metadata_extractor.detect_language(text)
        doc_type = self.metadata_extractor.classify_document_type(file_path.name, text)
        year = self.metadata_extractor.extract_year(text)

        logger.info(f"  Metadata: language={language}, type={doc_type}, year={year}")

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
    # Основная функция загрузки документов
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
