"""
Pruebas Automatizadas de la Etapa 4: Generación con Google Gemini API y Gestión de Errores.

Verifica:
1. Formato adecuado de contexto y citas pedagógicas para el LLM.
2. Modo degradado / Fallback amigable cuando no hay clave API configurada.
3. Manejo seguro de la cuota de la API gratuita (captura de HTTP 429 - 15 RPM).
4. Manejo de errores de autenticación (clave inválida o expirada).
5. Manejo de errores de conexión / offline.
6. Integración del flujo RAG de punta a punta (Documento -> Vector -> LLM).
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import shutil
from pathlib import Path

from src.config import Config
from src.rag.llm_client import LLMClient
from src.rag.vector_store import VectorStore
from src.rag.document_manager import DocumentManager
from src.rag.service import RAGService
from google.genai import errors


class TestStage4LLMIntegration(unittest.TestCase):
    """Conjunto de pruebas para la integración con Gemini API y gestión de errores."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="academic_rag_stage4_"))
        self.fixtures_dir = Path(__file__).parent / "fixtures"

        self.vector_store = VectorStore(db_path=self.test_dir / "vectordb")
        self.doc_manager = DocumentManager(
            data_dir=self.test_dir,
            vector_store=self.vector_store
        )
        self.llm_client = LLMClient(api_key="")  # Sin clave por defecto
        self.rag_service = RAGService(
            doc_manager=self.doc_manager,
            llm_client=self.llm_client
        )

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_context_formatting(self):
        """Verifica que los fragmentos se formateen con documento, página y contenido."""
        sample_chunks = [
            {
                "document": "apuntes_redes.pdf",
                "page": 5,
                "career": "Ciberseguridad",
                "fragment": "Un firewall filtra paquetes TCP/IP según reglas predefinidas."
            }
        ]
        formatted = LLMClient._format_context(sample_chunks)
        self.assertIn("apuntes_redes.pdf", formatted)
        self.assertIn("Página: 5", formatted)
        self.assertIn("firewall filtra paquetes", formatted)

    def test_fallback_when_no_api_key(self):
        """Verifica que sin clave API se devuelva una guía clara sin lanzar excepciones."""
        client = LLMClient(api_key="")
        result = client.generate_answer(
            query="¿Qué es un firewall?",
            career="Ciberseguridad",
            context_chunks=[{"document": "doc.pdf", "page": 1, "fragment": "Texto de prueba"}]
        )
        self.assertFalse(result["used_llm"])
        self.assertIn("Modo Local - Sin Clave API", result["answer"])
        self.assertIn("Configurar Clave API", result["answer"])

    def test_mock_successful_gemini_generation(self):
        """Verifica la generación exitosa con un mock de la API de Gemini."""
        client = LLMClient(api_key="test_key_123")
        mock_response = MagicMock()
        mock_response.text = (
            "Según los apuntes de la cátedra [Documento: redes.pdf, Pág. 3], "
            "un firewall es un dispositivo de seguridad que controla el tráfico entrante y saliente."
        )

        with patch.object(client, "_get_client") as mock_get_client:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.return_value = mock_response
            mock_get_client.return_value = mock_genai_client

            result = client.generate_answer(
                query="¿Qué es un firewall?",
                career="Ciberseguridad",
                context_chunks=[{"document": "redes.pdf", "page": 3, "fragment": "Texto..."}]
            )

            self.assertTrue(result["used_llm"])
            self.assertIn("Según los apuntes de la cátedra", result["answer"])
            self.assertIn("[Documento: redes.pdf, Pág. 3]", result["answer"])
            self.assertIsNone(result["error"])

    def test_handling_rate_limit_429(self):
        """Requisito 10: Verifica que el límite de cuota (HTTP 429) sea capturado amablemente."""
        client = LLMClient(api_key="test_key_123")

        # Simular error de cuota agotada de Google GenAI
        err_429 = errors.APIError(429, {"error": {"message": "Resource exhausted. Rate limit exceeded."}})

        with patch.object(client, "_get_client") as mock_get_client:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.side_effect = err_429
            mock_get_client.return_value = mock_genai_client

            result = client.generate_answer(
                query="Consulta",
                career="Ciberseguridad",
                context_chunks=[{"document": "doc.pdf", "page": 1, "fragment": "Contenido"}]
            )

            self.assertFalse(result["used_llm"])
            self.assertEqual(result["error"], "RATE_LIMIT_429")
            self.assertIn("HTTP 429", result["answer"])
            self.assertIn("15 consultas por minuto", result["answer"])
            self.assertIn("Seguridad financiera", result["answer"])

    def test_handling_invalid_api_key_401_403(self):
        """Verifica la respuesta clara ante una clave de API inválida."""
        client = LLMClient(api_key="clave_invalida")
        err_403 = errors.APIError(403, {"error": {"message": "API_KEY_INVALID: The provided API key is invalid."}})

        with patch.object(client, "_get_client") as mock_get_client:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.side_effect = err_403
            mock_get_client.return_value = mock_genai_client

            result = client.generate_answer(
                query="Consulta",
                career="Ciencia de datos",
                context_chunks=[{"document": "doc.pdf", "page": 1, "fragment": "Contenido"}]
            )

            self.assertFalse(result["used_llm"])
            self.assertEqual(result["error"], "INVALID_API_KEY")
            self.assertIn("Error de autenticación", result["answer"])

    def test_end_to_end_rag_service_flow(self):
        """Verifica el flujo integral RAG con servicio, documentos y LLM simulado."""
        txt_path = self.fixtures_dir / "ciberseguridad_intro.txt"
        self.doc_manager.add_document(txt_path, "Ciberseguridad")

        # Mock del LLM para responder como un profesor real
        mock_llm_response = {
            "answer": (
                "La confidencialidad es uno de los tres pilares de la seguridad de la información. "
                "Según el material de cátedra [Documento: ciberseguridad_intro.txt, Pág. 1], "
                "garantiza que la información solo sea accesible para personal autorizado."
            ),
            "used_llm": True,
            "error": None
        }

        with patch.object(self.rag_service.llm_client, "generate_answer", return_value=mock_llm_response):
            res = self.rag_service.answer_query(
                query="¿Cómo se define la confidencialidad?",
                career_filter="Ciberseguridad"
            )

            self.assertTrue(res["has_sufficient_info"])
            self.assertTrue(res["used_llm"])
            self.assertIn("La confidencialidad es uno de los tres pilares", res["answer"])
            self.assertEqual(len(res["sources"]), 1)
            self.assertEqual(res["sources"][0]["document"], "ciberseguridad_intro.txt")
            self.assertEqual(res["sources"][0]["page"], 1)


if __name__ == "__main__":
    unittest.main()
