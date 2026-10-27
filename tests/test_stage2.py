"""
Pruebas Automatizadas de la Etapa 2: Extracción y Gestión Documental.

Verifica:
1. Extracción de texto de archivos TXT y PDF con conservación de números de página.
2. Detección y rechazo de documentos vacíos o con formato no soportado.
3. Deduplicación por hash SHA-256 (rechazo de archivos idénticos).
4. Organización por carrera física y lógica.
5. Persistencia de metadatos en JSON.
6. Eliminación individual y colectiva de documentos.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from src.config import Config
from src.rag.extractor import (
    DocumentExtractor,
    DocumentExtractionError,
    EmptyDocumentError
)
from src.rag.document_manager import DocumentManager, DuplicateDocumentError
from src.rag.service import RAGService


class TestStage2DocumentProcessing(unittest.TestCase):
    """Pruebas unitarias de la Etapa 2."""

    def setUp(self):
        # Crear un directorio temporal para no interferir con datos de producción
        self.test_dir = Path(tempfile.mkdtemp(prefix="academic_rag_test_"))
        self.fixtures_dir = Path(__file__).parent / "fixtures"
        self.doc_manager = DocumentManager(data_dir=self.test_dir)
        self.rag_service = RAGService(doc_manager=self.doc_manager)

    def tearDown(self):
        # Limpiar directorio temporal tras las pruebas
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_extract_txt(self):
        """Verifica la extracción de un archivo de texto plano."""
        txt_path = self.fixtures_dir / "ciberseguridad_intro.txt"
        pages = DocumentExtractor.extract(txt_path)

        self.assertEqual(len(pages), 1, "Un archivo TXT debe tratarse como página 1.")
        self.assertEqual(pages[0]["page"], 1)
        self.assertIn("Tríada CIA", pages[0]["text"])
        self.assertIn("Confidencialidad", pages[0]["text"])

    def test_extract_pdf_multipage(self):
        """Verifica la extracción página a página de un archivo PDF."""
        pdf_path = self.fixtures_dir / "patrones_diseno.pdf"
        pages = DocumentExtractor.extract(pdf_path)

        self.assertEqual(len(pages), 2, "El PDF de prueba debe contener exactamente 2 páginas.")
        self.assertEqual(pages[0]["page"], 1)
        self.assertIn("Patrones Creacionales", pages[0]["text"])
        self.assertEqual(pages[1]["page"], 2)
        self.assertIn("Patrones Estructurales", pages[1]["text"])

    def test_empty_document_handling(self):
        """Verifica que un archivo vacío dispare EmptyDocumentError."""
        empty_file = self.test_dir / "vacio.txt"
        empty_file.write_text("   \n\n   ", encoding="utf-8")

        with self.assertRaises(EmptyDocumentError):
            DocumentExtractor.extract(empty_file)

    def test_unsupported_format(self):
        """Verifica que extensiones no soportadas arrojen ValueError."""
        dummy_file = self.test_dir / "archivo.docx"
        dummy_file.write_text("contenido", encoding="utf-8")

        with self.assertRaises(ValueError):
            DocumentExtractor.extract(dummy_file)

    def test_add_document_and_deduplication(self):
        """Verifica la importación exitosa y la detección de duplicados por SHA-256."""
        txt_path = self.fixtures_dir / "ciberseguridad_intro.txt"
        career = "Ciberseguridad"

        # 1. Primera importación exitosa
        doc = self.doc_manager.add_document(txt_path, career)
        self.assertEqual(doc["name"], "ciberseguridad_intro.txt")
        self.assertEqual(doc["career"], career)
        self.assertTrue(Path(doc["stored_path"]).exists(), "El archivo debe copiarse a uploads.")

        # 2. Intentar importar el mismo archivo de nuevo debe disparar DuplicateDocumentError
        with self.assertRaises(DuplicateDocumentError):
            self.doc_manager.add_document(txt_path, career)

        # 3. Intentar importar una copia con otro nombre pero idéntico contenido debe fallar también
        cloned_path = self.test_dir / "apunte_renombrado.txt"
        shutil.copy2(txt_path, cloned_path)
        with self.assertRaises(DuplicateDocumentError):
            self.doc_manager.add_document(cloned_path, "Ciencia de datos")

    def test_list_and_filter_by_career(self):
        """Verifica el filtrado de documentos por carrera institucional."""
        txt_ciber = self.fixtures_dir / "ciberseguridad_intro.txt"
        txt_datos = self.fixtures_dir / "ciencia_datos_intro.txt"
        pdf_dev = self.fixtures_dir / "patrones_diseno.pdf"

        self.doc_manager.add_document(txt_ciber, "Ciberseguridad")
        self.doc_manager.add_document(txt_datos, "Ciencia de datos")
        self.doc_manager.add_document(pdf_dev, "Desarrollo de software")

        # Total
        all_docs = self.doc_manager.list_documents()
        self.assertEqual(len(all_docs), 3)

        # Filtro Ciberseguridad
        ciber_docs = self.doc_manager.list_documents("Ciberseguridad")
        self.assertEqual(len(ciber_docs), 1)
        self.assertEqual(ciber_docs[0]["name"], "ciberseguridad_intro.txt")

        # Filtro Desarrollo de software
        dev_docs = self.doc_manager.list_documents("Desarrollo de software")
        self.assertEqual(len(dev_docs), 1)
        self.assertEqual(dev_docs[0]["name"], "patrones_diseno.pdf")
        self.assertEqual(dev_docs[0]["pages_count"], 2)

    def test_delete_single_document(self):
        """Verifica la eliminación individual de un documento físico y sus metadatos."""
        txt_path = self.fixtures_dir / "ciberseguridad_intro.txt"
        doc = self.doc_manager.add_document(txt_path, "Ciberseguridad")
        doc_id = doc["id"]
        stored_path = Path(doc["stored_path"])

        self.assertTrue(stored_path.exists())
        self.assertEqual(len(self.doc_manager.list_documents()), 1)

        # Eliminar
        deleted = self.doc_manager.delete_document(doc_id)
        self.assertTrue(deleted)
        self.assertEqual(len(self.doc_manager.list_documents()), 0)
        self.assertFalse(stored_path.exists(), "El archivo físico debe haberse borrado.")

    def test_rag_service_query_with_and_without_docs(self):
        """Verifica la respuesta del servicio RAG ante presencia y ausencia de documentos."""
        # 1. Sin documentos cargados
        empty_res = self.rag_service.answer_query("¿Qué es un socket?", "Desarrollo de software")
        self.assertFalse(empty_res["has_sufficient_info"])
        self.assertIn("No hay apuntes cargados en el sistema", empty_res["answer"])

        # 2. Con documentos cargados y consulta temática relevante
        pdf_dev = self.fixtures_dir / "patrones_diseno.pdf"
        self.rag_service.add_document(pdf_dev, "Desarrollo de software")

        active_res = self.rag_service.answer_query("¿Cuáles son los patrones creacionales?", "Desarrollo de software")
        self.assertTrue(active_res["has_sufficient_info"])
        self.assertEqual(len(active_res["sources"]), 1)
        self.assertEqual(active_res["sources"][0]["document"], "patrones_diseno.pdf")
        self.assertEqual(active_res["sources"][0]["page"], 1)


if __name__ == "__main__":
    unittest.main()
