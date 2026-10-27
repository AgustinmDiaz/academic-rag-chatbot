"""
Pruebas Automatizadas de la Etapa 1.

Verifica:
1. Resolución correcta de rutas según el sistema operativo.
2. Inicialización de la configuración y carpetas requeridas.
3. Comportamiento del servicio simulado RAG.
4. Creación y renderizado de la interfaz gráfica sin bloqueos.
"""

import sys
import unittest
from pathlib import Path

from src.utils.paths import get_bundle_dir, get_app_data_dir, is_frozen
from src.config import Config
from src.rag.service import RAGService
from src.gui.app import MainWindow


class TestStage1(unittest.TestCase):
    """Conjunto de pruebas para validar la Etapa 1."""

    def test_paths_resolution(self):
        """Verifica que las rutas base y de datos se resuelvan correctamente."""
        bundle_dir = get_bundle_dir()
        self.assertTrue(bundle_dir.exists(), "El directorio base del proyecto debe existir.")
        self.assertFalse(is_frozen(), "En ejecución de desarrollo no debe estar congelado.")

        app_data = get_app_data_dir("AcademicRAGTest")
        self.assertTrue(app_data.exists(), "El directorio de datos debe existir o haberse creado.")

    def test_config_and_careers(self):
        """Verifica que las 3 carreras de la institución estén configuradas."""
        self.assertEqual(len(Config.CAREERS), 3, "Deben existir exactamente 3 carreras configuradas.")
        self.assertIn("Ciberseguridad", Config.CAREERS)
        self.assertIn("Ciencia de datos", Config.CAREERS)
        self.assertIn("Desarrollo de software", Config.CAREERS)

        Config.initialize_directories()
        self.assertTrue(Config.UPLOADS_DIR.exists(), "La carpeta uploads debe crearse.")
        self.assertTrue(Config.VECTORDB_DIR.exists(), "La carpeta vectordb debe crearse.")

    def test_rag_service_mock(self):
        """Verifica que el servicio RAG responda y soporte filtrado por carrera."""
        service = RAGService()
        
        # Si la base local está vacía, agregar un apunte de prueba
        fixture_path = Path(__file__).parent / "fixtures" / "ciberseguridad_intro.txt"
        if not service.list_documents() and fixture_path.exists():
            service.add_document(fixture_path, "Ciberseguridad")

        # Listar documentos
        all_docs = service.list_documents()
        self.assertGreaterEqual(len(all_docs), 1)

        # Filtrar por carrera específica
        it_docs = service.list_documents(career_filter="Ciberseguridad")
        self.assertTrue(all(d["career"] == "Ciberseguridad" for d in it_docs))

        # Realizar consulta de prueba
        res = service.answer_query("¿Qué es la confidencialidad?", "Ciberseguridad")
        self.assertIn("answer", res)
        self.assertIn("sources", res)
        self.assertIn("has_sufficient_info", res)

    def test_gui_initialization(self):
        """Verifica que la ventana principal de CustomTkinter pueda crearse, renderizarse y destruirse."""
        app = MainWindow()
        # Forzar un ciclo de actualización de eventos de la GUI
        app.update()

        # Comprobar componentes clave
        self.assertIsNotNone(app.sidebar_frame)
        self.assertIsNotNone(app.chat_container)
        self.assertIsNotNone(app.sources_frame)
        self.assertIsNotNone(app.status_bar)

        # Limpiar
        app.destroy()


if __name__ == "__main__":
    unittest.main()
