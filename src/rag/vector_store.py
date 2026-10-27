"""
Módulo de Almacenamiento y Búsqueda Vectorial con ChromaDB.

Gestiona:
1. La persistencia local de la base de datos vectorial en disco.
2. La generación automática de embeddings mediante el modelo ONNX MiniLM local.
3. El filtrado de búsqueda semántica por carrera institucional.
4. La eliminación sincronizada de fragmentos al borrar un documento.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions

from src.config import Config


class VectorStore:
    """Encapsula la colección vectorial local de ChromaDB."""

    COLLECTION_NAME = "academic_notes"

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Config.VECTORDB_DIR
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Cliente persistente local en disco
        self.client = chromadb.PersistentClient(path=str(self.db_path))

        # Función de embeddings local mediante ONNX (384 dimensiones)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Obtener o crear colección con distancia coseno
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Calcula embeddings y almacena los fragmentos en la base vectorial con sus metadatos.
        Retorna la cantidad de fragmentos insertados.
        """
        if not chunks:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for c in chunks:
            ids.append(c["id"])
            documents.append(c["text"])
            metadatas.append({
                "doc_id": c["doc_id"],
                "document": c["document"],
                "career": c["career"],
                "page": int(c["page"]),
                "chunk_index": int(c["chunk_index"])
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        return len(chunks)

    def search(
        self,
        query: str,
        career_filter: Optional[str] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Busca los fragmentos más relevantes semánticamente para la pregunta.
        Aplica filtro estricto por carrera institucional si no es 'Todas las carreras'.
        """
        query_text = query.strip()
        if not query_text or self.count() == 0:
            return []

        # Configurar filtro WHERE si se especificó una carrera concreta
        where_filter = None
        if career_filter and career_filter != Config.ALL_CAREERS_LABEL:
            where_filter = {"career": career_filter}

        # Asegurar que top_k no supere el total de elementos indexados
        total_items = self.count()
        n_results = min(top_k, total_items)
        if n_results <= 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter
            )
        except Exception as e:
            print(f"[VectorStore] Error en consulta vectorial: {e}")
            return []

        formatted_matches: List[Dict[str, Any]] = []
        
        # Validar si hubo resultados
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids_list = results["ids"][0]
        docs_list = results["documents"][0] if results.get("documents") else []
        metas_list = results["metadatas"][0] if results.get("metadatas") else []
        distances_list = results["distances"][0] if results.get("distances") else []

        for idx in range(len(ids_list)):
            meta = metas_list[idx] if idx < len(metas_list) else {}
            dist = distances_list[idx] if idx < len(distances_list) else 1.0
            
            # En distancia coseno de Chroma: distancia = 1 - similitud_coseno.
            # Por tanto: similitud = 1 - distancia
            similarity = max(0.0, min(1.0, 1.0 - dist))

            formatted_matches.append({
                "id": ids_list[idx],
                "document": meta.get("document", "Desconocido"),
                "career": meta.get("career", "N/A"),
                "page": meta.get("page", 1),
                "fragment": docs_list[idx] if idx < len(docs_list) else "",
                "distance": dist,
                "similarity": round(similarity, 4)
            })

        return formatted_matches

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Elimina todos los fragmentos asociados a un documento específico.
        """
        try:
            # Obtener IDs antes de borrar para contar cuántos se eliminaron
            existing = self.collection.get(where={"doc_id": doc_id})
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception as e:
            print(f"[VectorStore] Error al eliminar fragmentos de doc_id {doc_id}: {e}")
            return 0

    def count(self) -> int:
        """Retorna la cantidad total de fragmentos indexados."""
        return self.collection.count()

    def clear_all(self) -> None:
        """Elimina todos los fragmentos de la colección."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
