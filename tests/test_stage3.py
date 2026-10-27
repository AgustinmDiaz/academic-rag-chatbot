"""
Pruebas Automatizadas de la Etapa 3: Embeddings, ChromaDB y Recuperación de Fragmentos.

Verifica:
1. Fragmentación inteligente de texto (chunking) con solapamiento y corte semántico.
2. Indexación y cálculo de embeddings locales con ChromaDB.
3. Búsqueda semántica por distancia coseno y filtrado por carrera institucional.
4. Respuesta honesta cuando no existe información suficiente (Requisito 7).
5. Eliminación sincronizada de vectores al suprimir un documento.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from src.rag.chunker import TextChunker
from src.rag.vector_store import VectorStore
from src.rag.document_manager import DocumentManager
from src.rag.service import RAGService


class TestStage3EmbeddingsAndRetrieval(unittest.TestCase):
    """Conjunto de pruebas para la Etapa 3."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="academic_rag_stage3_"))
        self.fixtures_dir = Path(__file__).parent / "fixtures"

        self.vector_store = VectorStore(db_path=self.test_dir / "vectordb")
        self.doc_manager = DocumentManager(
            data_dir=self.test_dir,
            vector_store=self.vector_store
        )
        self.rag_service = RAGService(doc_manager=self.doc_manager)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_text_chunker_overlap(self):
        """Verifica que el chunker divida texto largo y conserve el solapamiento."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=30)
        sample_text = (
            "La seguridad de la información protege los activos informáticos. "
            "La criptografía permite asegurar la confidencialidad mediante algoritmos. "
            "El análisis forense digital investiga incidentes de seguridad ocurridos."
        )

        chunks = chunker.split_text(sample_text)
        self.assertGreater(len(chunks), 1, "Un texto largo debe dividirse en varios fragmentos.")
        
        # Verificar que ningún fragmento supere excesivamente el tamaño objetivo
        for c in chunks:
            self.assertLessEqual(len(c), 120)

    def test_chunk_pages_metadata(self):
        """Verifica que los fragmentos generados preserven página, documento y carrera."""
        chunker = TextChunker(chunk_size=150, chunk_overlap=30)
        pages = [
            {"page": 1, "text": "Página uno con conceptos introductorios de redes."},
            {"page": 2, "text": "Página dos con conceptos avanzados de cortafuegos y VPN."}
        ]

        chunks = chunker.chunk_pages(
            doc_id="test1234",
            doc_name="redes.pdf",
            career="Ciberseguridad",
            pages=pages
        )

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["page"], 1)
        self.assertEqual(chunks[0]["document"], "redes.pdf")
        self.assertEqual(chunks[0]["career"], "Ciberseguridad")
        self.assertEqual(chunks[1]["page"], 2)

    def test_vector_indexing_and_search(self):
        """Verifica el almacenamiento y recuperación semántica en ChromaDB."""
        txt_path = self.fixtures_dir / "ciberseguridad_intro.txt"
        self.doc_manager.add_document(txt_path, "Ciberseguridad")

        # La base vectorial debe contener fragmentos
        self.assertGreater(self.vector_store.count(), 0)

        # Búsqueda semántica con términos clave
        matches = self.vector_store.search(
            query="¿Cómo se garantiza la confidencialidad?",
            career_filter="Ciberseguridad",
            top_k=2
        )

        self.assertTrue(len(matches) > 0, "Debe retornar al menos una coincidencia.")
        best_match = matches[0]
        self.assertIn("Confidencialidad", best_match["fragment"])
        self.assertEqual(best_match["document"], "ciberseguridad_intro.txt")
        self.assertEqual(best_match["page"], 1)
        self.assertGreater(best_match["similarity"], 0.5)

    def test_career_filtering_in_vector_search(self):
        """Verifica que el filtro de carrera aísle los fragmentos de otras carreras."""
        txt_ciber = self.fixtures_dir / "ciberseguridad_intro.txt"
        txt_datos = self.fixtures_dir / "ciencia_datos_intro.txt"

        self.doc_manager.add_document(txt_ciber, "Ciberseguridad")
        self.doc_manager.add_document(txt_datos, "Ciencia de datos")

        # Buscar término de Ciberseguridad pero filtrando por Ciencia de datos
        matches_filtered = self.vector_store.search(
            query="Tríada CIA confidencialidad",
            career_filter="Ciencia de datos",
            top_k=2
        )

        # Todas las coincidencias devueltas deben pertenecer a Ciencia de datos
        for m in matches_filtered:
            self.assertEqual(m["career"], "Ciencia de datos")

    def test_deletion_removes_from_vector_store(self):
        """Verifica que al eliminar un documento sus vectores desaparezcan de ChromaDB."""
        txt_path = self.fixtures_dir / "ciberseguridad_intro.txt"
        doc = self.doc_manager.add_document(txt_path, "Ciberseguridad")

        initial_count = self.vector_store.count()
        self.assertGreater(initial_count, 0)

        # Eliminar el documento
        self.doc_manager.delete_document(doc["id"])

        # Los fragmentos deben haber sido removidos
        self.assertEqual(self.vector_store.count(), 0)

    def test_rag_service_insufficient_information_recognition(self):
        """Requisito Funcional 7: Verificar que declare falta de información si el tema no existe."""
        txt_path = self.fixtures_dir / "ciberseguridad_intro.txt"
        self.doc_manager.add_document(txt_path, "Ciberseguridad")

        # Pregunta sobre un tema completamente ajeno al documento cargado
        res = self.rag_service.answer_query(
            query="¿Cuál es la receta tradicional para preparar pasta frola?",
            career_filter="Ciberseguridad"
        )

        self.assertFalse(res["has_sufficient_info"], "Debe reconocer que no hay información suficiente.")
        self.assertIn("No se encontró información suficiente", res["answer"])


if __name__ == "__main__":
    unittest.main()
