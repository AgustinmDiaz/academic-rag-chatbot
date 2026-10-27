"""Módulo RAG (Recuperación y Generación Aumentada)."""
from .service import RAGService
from .document_manager import DocumentManager, DuplicateDocumentError
from .extractor import (
    DocumentExtractor,
    DocumentExtractionError,
    EmptyDocumentError,
    EncryptedPDFError,
    OCREngineNotFoundError
)
from .chunker import TextChunker
from .vector_store import VectorStore
from .llm_client import LLMClient
from .citation_verifier import CitationVerifier

__all__ = [
    "RAGService",
    "DocumentManager",
    "DuplicateDocumentError",
    "DocumentExtractor",
    "DocumentExtractionError",
    "EmptyDocumentError",
    "EncryptedPDFError",
    "OCREngineNotFoundError",
    "TextChunker",
    "VectorStore",
    "LLMClient",
    "CitationVerifier",
]
