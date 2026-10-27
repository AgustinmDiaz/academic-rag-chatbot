"""
Módulo de Servicio RAG.

Coordina:
1. La gestión documental y extracción (DocumentManager).
2. La fragmentación (TextChunker).
3. La búsqueda semántica vectorial (VectorStore).
4. La generación de respuestas y gestión de cuotas de la API (LLMClient).
5. La verificación de citas y detección de alucinaciones (CitationVerifier).
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import Config
from src.rag.document_manager import DocumentManager, DuplicateDocumentError
from src.rag.vector_store import VectorStore
from src.rag.chunker import TextChunker
from src.rag.extractor import DocumentExtractionError, EmptyDocumentError, EncryptedPDFError
from src.rag.llm_client import LLMClient
from src.rag.citation_verifier import CitationVerifier


class RAGService:
    """Servicio que orquesta el flujo RAG completo con verificación de citas y fidelidad."""

    # Umbral mínimo de similitud para considerar que hay evidencia relevante (0.0 a 1.0)
    SIMILARITY_THRESHOLD = 0.40

    def __init__(
        self,
        doc_manager: Optional[DocumentManager] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.doc_manager = doc_manager or DocumentManager()
        self.llm_client = llm_client or LLMClient()

    def list_documents(self, career_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retorna la lista de documentos registrados, opcionalmente filtrados por carrera."""
        return self.doc_manager.list_documents(career_filter=career_filter)

    def add_document(self, file_path: Path, career: str) -> Dict[str, Any]:
        """Importa un documento real, extrae su texto y genera sus embeddings en ChromaDB."""
        return self.doc_manager.add_document(file_path, career)

    def delete_document(self, doc_id: str) -> bool:
        """Elimina un documento y purga sus fragmentos de la base vectorial."""
        return self.doc_manager.delete_document(doc_id)

    def clear_documents(self) -> int:
        """Elimina todos los documentos cargados y vacía la base vectorial."""
        return self.doc_manager.clear_all()

    def answer_query(self, query: str, career_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Flujo RAG completo con verificación de cero alucinaciones:
        1. Valida existencia de documentos en la carrera seleccionada.
        2. Realiza búsqueda semántica vectorial en ChromaDB.
        3. Evalúa si la similitud supera el umbral de evidencia suficiente (Requisito 7).
        4. Si hay evidencia suficiente, invoca a Google Gemini con prompt estricto (Requisito 5).
        5. Verifica la correspondencia de citas emitidas contra los fragmentos reales (Requisito 6).
        6. Devuelve la respuesta fundamentada con citas verificadas y fuentes detalladas.
        """
        docs = self.list_documents(career_filter=career_filter)
        filter_name = career_filter if career_filter and career_filter != Config.ALL_CAREERS_LABEL else "todas las carreras"

        # 1. Verificar si existen documentos para la carrera seleccionada
        if not docs:
            return {
                "answer": (
                    f"⚠️ No hay apuntes cargados en el sistema para {filter_name}.\n\n"
                    f"Para poder responder sobre '{query}', por favor importa materiales académicos (PDF o TXT) "
                    f"utilizando el botón '📥 Importar' de la barra lateral."
                ),
                "sources": [],
                "has_sufficient_info": False,
                "used_llm": False,
                "verification": {"is_fully_grounded": True, "total_citations": 0}
            }

        # 2. Búsqueda vectorial en ChromaDB
        matches = self.doc_manager.vector_store.search(
            query=query,
            career_filter=career_filter,
            top_k=4
        )

        # 3. Validar si se encontraron fragmentos con suficiente relevancia
        relevant_matches = [m for m in matches if m.get("similarity", 0) >= self.SIMILARITY_THRESHOLD]

        if not relevant_matches:
            return {
                "answer": (
                    f"No se encontró información suficiente en los materiales cargados en el sistema sobre:\n"
                    f"\"{query}\" para la carrera {filter_name}.\n\n"
                    f"💡 Sugerencia pedagógica: Intenta reformular tu pregunta con otros términos o "
                    f"comprueba si el tema corresponde al programa de la carrera seleccionada."
                ),
                "sources": matches[:1] if matches else [],
                "has_sufficient_info": False,
                "used_llm": False,
                "verification": {"is_fully_grounded": True, "total_citations": 0}
            }

        # 4. Formatear fuentes y citas para la interfaz gráfica
        sources = []
        for m in relevant_matches:
            sources.append({
                "document": m["document"],
                "career": m["career"],
                "page": m["page"],
                "fragment": m["fragment"],
                "similarity": m.get("similarity", 0.0)
            })

        # 5. Generar respuesta sintetizada mediante LLMClient (Google Gemini API)
        assigned_career = relevant_matches[0]["career"] if relevant_matches else filter_name
        llm_response = self.llm_client.generate_answer(
            query=query,
            career=assigned_career,
            context_chunks=relevant_matches
        )

        answer_text = llm_response.get("answer", "No se generó respuesta.")

        # Detectar si la respuesta del modelo concluye falta de información suficiente
        negative_markers = [
            "no contienen información suficiente",
            "no contiene información suficiente",
            "no se encontró información suficiente",
            "no contiene información sobre",
            "no contienen información sobre"
        ]
        answer_lower = answer_text.lower()
        has_sufficient = not any(m in answer_lower for m in negative_markers)

        # 6. Verificación de citas y CERO alucinaciones (Etapa 5)
        verification = CitationVerifier.verify_citations(
            answer_text=answer_text,
            retrieved_chunks=relevant_matches
        )

        return {
            "answer": answer_text,
            "sources": sources,
            "has_sufficient_info": has_sufficient,
            "used_llm": llm_response.get("used_llm", False),
            "error": llm_response.get("error"),
            "verification": verification
        }
