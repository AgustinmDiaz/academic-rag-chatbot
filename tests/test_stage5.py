"""
Pruebas Automatizadas de la Etapa 5: Citas, Fuentes y Verificación de Cero Alucinaciones.

Verifica:
1. Extracción y análisis de citas emitidas por el modelo de lenguaje.
2. Detección y bloqueo de citas inventadas o no fundamentadas en los fragmentos.
3. Comportamiento estricto ante preguntas capciosas o con premisas falsas (Cero Alucinaciones).
4. Declaración honesta de información insuficiente cuando el tema no figura en los apuntes (Requisito 7).
5. Visualización y consistencia de las fuentes en la interfaz gráfica.
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import shutil
from pathlib import Path

from src.rag.citation_verifier import CitationVerifier
from src.rag.vector_store import VectorStore
from src.rag.document_manager import DocumentManager
from src.rag.service import RAGService
from src.gui.app import MainWindow


class TestStage5CitationsAndZeroHallucinations(unittest.TestCase):
    """Pruebas de verificación de citas y rigurosidad documental."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="academic_rag_stage5_"))
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

    def test_citation_extraction_patterns(self):
        """Verifica la detección de citas en múltiples formatos estándar."""
        text_1 = "La confidencialidad es clave [Documento: ciber_intro.txt, Pág. 1] y se complementa con la integridad [Doc: ciber_intro.txt, Pag. 2]."
        citations_1 = CitationVerifier.extract_citations(text_1)
        self.assertEqual(len(citations_1), 2)
        self.assertEqual(citations_1[0]["page"], 1)
        self.assertEqual(citations_1[1]["page"], 2)

        text_2 = "El patrón Factory se explica en [Pág. 1 - patrones_diseno.pdf]."
        citations_2 = CitationVerifier.extract_citations(text_2)
        self.assertEqual(len(citations_2), 1)
        self.assertEqual(citations_2[0]["document"], "patrones_diseno.pdf")
        self.assertEqual(citations_2[0]["page"], 1)

    def test_citation_verifier_perfect_grounding(self):
        """Verifica que citas 100% auténticas obtengan score 1.0."""
        retrieved_chunks = [
            {"document": "ciberseguridad_intro.txt", "page": 1, "fragment": "La confidencialidad protege..."}
        ]
        answer = "Según los apuntes [Documento: ciberseguridad_intro.txt, Pág. 1], la confidencialidad es esencial."

        res = CitationVerifier.verify_citations(answer, retrieved_chunks)
        self.assertTrue(res["is_fully_grounded"])
        self.assertEqual(res["fidelity_score"], 1.0)
        self.assertEqual(len(res["valid_citations"]), 1)
        self.assertEqual(len(res["hallucinated_citations"]), 0)

    def test_citation_verifier_detects_hallucination(self):
        """Verifica que el sistema detecte citas a documentos o páginas inexistentes."""
        retrieved_chunks = [
            {"document": "ciberseguridad_intro.txt", "page": 1, "fragment": "La confidencialidad protege..."}
        ]
        # El modelo cita erróneamente un libro de física que nunca estuvo en el contexto
        hallucinated_answer = (
            "Según [Documento: fisica_cuantica.pdf, Pág. 42], las partículas subatómicas son fundamentales. "
            "Además [Documento: ciberseguridad_intro.txt, Pág. 1], la confidencialidad es clave."
        )

        res = CitationVerifier.verify_citations(hallucinated_answer, retrieved_chunks)
        self.assertFalse(res["is_fully_grounded"], "Debe detectar que hay al menos una cita falsa.")
        self.assertEqual(len(res["valid_citations"]), 1)
        self.assertEqual(len(res["hallucinated_citations"]), 1)
        self.assertEqual(res["hallucinated_citations"][0]["document"], "fisica_cuantica.pdf")
        self.assertEqual(res["fidelity_score"], 0.5)

    def test_zero_hallucination_false_premise_rejection(self):
        """Verifica el rechazo a responder premisas falsas no contenidas en los apuntes."""
        txt_path = self.fixtures_dir / "ciberseguridad_intro.txt"
        self.doc_manager.add_document(txt_path, "Ciberseguridad")

        # Mock del LLM respondiendo que los apuntes no contienen información sobre la premisa falsa
        mock_rejection = {
            "answer": (
                "Los materiales cargados en el sistema no contienen información sobre telepatía "
                "ni viajes en el tiempo en la Tríada CIA. La Tríada CIA solo comprende Confidencialidad, "
                "Integridad y Disponibilidad [Documento: ciberseguridad_intro.txt, Pág. 1]."
            ),
            "used_llm": True,
            "error": None
        }

        with patch.object(self.rag_service.llm_client, "generate_answer", return_value=mock_rejection):
            res = self.rag_service.answer_query(
                query="¿Por qué la Tríada CIA incluye la telepatía y los viajes en el tiempo según los apuntes?",
                career_filter="Ciberseguridad"
            )

            # Debe detectar el marcador negativo y declarar falta de información para la premisa falsa
            self.assertFalse(res["has_sufficient_info"])
            self.assertIn("no contienen información sobre telepatía", res["answer"].lower())

    def test_zero_hallucination_out_of_catalog_rejection(self):
        """Verifica que preguntas completamente fuera de contexto académico sean rechazadas."""
        txt_path = self.fixtures_dir / "ciencia_datos_intro.txt"
        self.doc_manager.add_document(txt_path, "Ciencia de datos")

        res = self.rag_service.answer_query(
            query="¿Quién ganó el torneo mundial de ajedrez en 1972?",
            career_filter="Ciencia de datos"
        )

        self.assertFalse(res["has_sufficient_info"])
        self.assertIn("No se encontró información suficiente", res["answer"])

    def test_gui_sources_panel_rendering_with_badges(self):
        """Verifica que la interfaz gráfica renderice correctamente las tarjetas de citas con porcentajes."""
        app = MainWindow(rag_service=self.rag_service)
        app.update()

        sources = [
            {
                "document": "patrones_diseno.pdf",
                "career": "Desarrollo de software",
                "page": 1,
                "fragment": "El patrón Singleton garantiza una única instancia.",
                "similarity": 0.88
            }
        ]
        verification = {
            "is_fully_grounded": True,
            "has_citations": True,
            "total_citations": 1
        }

        app._display_sources(sources, verification=verification)
        app.update()

        # Verificar que se crearon los widgets de la tarjeta
        content_children = app.sources_content.winfo_children()
        self.assertGreater(len(content_children), 0)

        app.destroy()


if __name__ == "__main__":
    unittest.main()
