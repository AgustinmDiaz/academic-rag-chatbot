"""
Módulo de Gestión y Persistencia de Documentos Académicos.

Se encarga de:
1. Validar la carrera de cada documento.
2. Calcular el hash SHA-256 del contenido para evitar duplicados.
3. Almacenar copias organizadas en data/uploads/<carrera>/.
4. Extraer el texto mediante DocumentExtractor.
5. Fragmentar el texto en fragmentos con solapamiento mediante TextChunker.
6. Indexar los fragmentos en la base de datos vectorial ChromaDB.
7. Persistir los metadatos y páginas extraídas en documents_meta.json.
8. Permitir listar, consultar y eliminar documentos sincronizadamente.
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import Config
from src.rag.extractor import DocumentExtractor, DocumentExtractionError
from src.rag.chunker import TextChunker
from src.rag.vector_store import VectorStore


class DuplicateDocumentError(Exception):
    """El documento ya ha sido importado previamente (mismo contenido SHA-256)."""
    pass


class DocumentManager:
    """Administrador central de documentos y coordinación vectorial."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.data_dir = data_dir or Config.DATA_DIR
        self.uploads_dir = self.data_dir / "uploads"
        self.meta_file = self.data_dir / "documents_meta.json"

        # Inicializar componentes de fragmentación y almacenamiento vectorial
        self.chunker = TextChunker(chunk_size=600, chunk_overlap=120)
        self.vector_store = vector_store or VectorStore(db_path=self.data_dir / "vectordb")

        # Asegurar directorios
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_meta_file()

    def _ensure_meta_file() -> None:
        pass

    def _ensure_meta_file(self) -> None:
        """Crea el archivo JSON de metadatos si no existe."""
        if not self.meta_file.exists():
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)

    def _load_meta(self) -> List[Dict[str, Any]]:
        """Carga la lista de metadatos desde el archivo JSON."""
        try:
            with open(self.meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[DocumentManager] Error al cargar metadatos: {e}")
            return []

    def _save_meta(self, meta_list: List[Dict[str, Any]]) -> None:
        """Guarda la lista de metadatos en el archivo JSON."""
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, indent=2, ensure_ascii=False)

    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        """Calcula el hash SHA-256 del contenido de un archivo en bloques de 64 KB."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def add_document(self, source_path: Path, career: str) -> Dict[str, Any]:
        """
        Importa un documento PDF o TXT al sistema institucional:
        - Valida extensión y existencia.
        - Calcula hash SHA-256 para evitar duplicaciones.
        - Extrae texto página por página.
        - Almacena una copia en data/uploads/<carrera>/.
        - Genera fragmentos solapados y los indexa en ChromaDB.
        - Registra metadatos completos.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"El archivo origen no existe: {source}")

        if career not in Config.CAREERS and career != Config.ALL_CAREERS_LABEL:
            raise ValueError(f"Carrera no válida: '{career}'. Debe ser una de: {Config.CAREERS}")

        assigned_career = Config.CAREERS[0] if career == Config.ALL_CAREERS_LABEL else career

        # 1. Comprobar duplicados por hash SHA-256
        file_hash = self.calculate_sha256(source)
        meta_list = self._load_meta()

        for doc in meta_list:
            if doc.get("sha256") == file_hash:
                raise DuplicateDocumentError(
                    f"El archivo '{source.name}' ya fue importado anteriormente con el nombre "
                    f"'{doc.get('name')}' para la carrera '{doc.get('career')}'."
                )

        # 2. Extraer texto y validar legibilidad
        extracted_pages = DocumentExtractor.extract(source)

        # 3. Copiar archivo a la carpeta institucional
        career_dir = self.uploads_dir / assigned_career.replace(" ", "_").lower()
        career_dir.mkdir(parents=True, exist_ok=True)

        target_file_path = career_dir / source.name
        if target_file_path.exists():
            stem = source.stem
            suffix = source.suffix
            counter = 1
            while target_file_path.exists():
                target_file_path = career_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.copy2(source, target_file_path)

        # 4. Fragmentación e indexación en base vectorial
        doc_id = file_hash[:12]
        chunks = self.chunker.chunk_pages(
            doc_id=doc_id,
            doc_name=source.name,
            career=assigned_career,
            pages=extracted_pages
        )

        indexed_count = self.vector_store.add_chunks(chunks)

        # 5. Generar y guardar metadatos
        doc_entry = {
            "id": doc_id,
            "name": source.name,
            "stored_name": target_file_path.name,
            "career": assigned_career,
            "extension": source.suffix.lower(),
            "sha256": file_hash,
            "size_bytes": source.stat().st_size,
            "pages_count": len(extracted_pages),
            "total_chars": sum(p["char_count"] for p in extracted_pages),
            "chunks_count": indexed_count,
            "stored_path": str(target_file_path),
            "created_at": datetime.now().isoformat(),
            "extracted_pages": extracted_pages
        }

        meta_list.append(doc_entry)
        self._save_meta(meta_list)

        return doc_entry

    def list_documents(self, career_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retorna los documentos registrados, opcionalmente filtrados por carrera."""
        meta_list = self._load_meta()
        if not career_filter or career_filter == Config.ALL_CAREERS_LABEL:
            return meta_list
        return [doc for doc in meta_list if doc.get("career") == career_filter]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene la información completa de un documento por su ID."""
        for doc in self._load_meta():
            if doc.get("id") == doc_id:
                return doc
        return None

    def delete_document(self, doc_id: str) -> bool:
        """
        Elimina un documento:
        - Borra sus fragmentos y embeddings de ChromaDB.
        - Borra el archivo físico en disco.
        - Elimina el registro del archivo de metadatos.
        """
        meta_list = self._load_meta()
        target_doc = None
        new_list = []

        for doc in meta_list:
            if doc.get("id") == doc_id:
                target_doc = doc
            else:
                new_list.append(doc)

        if not target_doc:
            return False

        # 1. Eliminar fragmentos de ChromaDB
        self.vector_store.delete_by_doc_id(doc_id)

        # 2. Eliminar archivo físico
        try:
            stored_path = Path(target_doc.get("stored_path", ""))
            if stored_path.exists():
                stored_path.unlink()
        except Exception as e:
            print(f"[DocumentManager] Advertencia al borrar archivo: {e}")

        # 3. Guardar metadatos actualizados
        self._save_meta(new_list)
        return True

    def clear_all(self) -> int:
        """Elimina todos los documentos y vacía la base vectorial."""
        meta_list = self._load_meta()
        count = len(meta_list)

        # Vaciar colección vectorial
        self.vector_store.clear_all()

        # Eliminar archivos físicos
        for doc in meta_list:
            try:
                stored_path = Path(doc.get("stored_path", ""))
                if stored_path.exists():
                    stored_path.unlink()
            except Exception:
                pass

        self._save_meta([])
        return count
