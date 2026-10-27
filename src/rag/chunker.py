"""
Módulo de Fragmentación de Texto (Chunking) para Documentos Académicos.

Divide los textos de cada página en fragmentos continuos con solapamiento (overlap),
respetando saltos de línea y oraciones para preservar el sentido pedagógico de los conceptos.
"""

from typing import List, Dict, Any
import re


class TextChunker:
    """Estrategia de fragmentación basada en caracteres con solapamiento."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Divide un texto extenso en fragmentos más pequeños solapados.
        Prioriza cortes en párrafos o puntos seguidos para no romper oraciones.
        """
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size

            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Buscar el mejor punto de corte hacia atrás (párrafo, punto, salto de línea o espacio)
            cut = -1
            search_window = text[start:end]

            # 1. Intentar cortar en salto de párrafo
            p_idx = search_window.rfind("\n\n")
            if p_idx != -1 and p_idx > self.chunk_overlap:
                cut = start + p_idx + 2
            else:
                # 2. Intentar cortar en punto y espacio
                dot_idx = search_window.rfind(". ")
                if dot_idx != -1 and dot_idx > self.chunk_overlap:
                    cut = start + dot_idx + 2
                else:
                    # 3. Intentar cortar en salto de línea simple
                    nl_idx = search_window.rfind("\n")
                    if nl_idx != -1 and nl_idx > self.chunk_overlap:
                        cut = start + nl_idx + 1
                    else:
                        # 4. Intentar cortar en espacio en blanco
                        sp_idx = search_window.rfind(" ")
                        if sp_idx != -1 and sp_idx > self.chunk_overlap:
                            cut = start + sp_idx + 1
                        else:
                            # 5. Corte forzado en chunk_size
                            cut = end

            chunk = text[start:cut].strip()
            if chunk:
                chunks.append(chunk)

            # Avanzar la ventana restando el solapamiento para mantener el contexto
            start = max(cut - self.chunk_overlap, start + 1)

        return chunks

    def chunk_pages(
        self,
        doc_id: str,
        doc_name: str,
        career: str,
        pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convierte una lista de páginas extraídas en fragmentos estructurados con metadatos.
        Garantiza que cada fragmento conozca su documento de origen, carrera y número de página.
        """
        all_chunks: List[Dict[str, Any]] = []
        chunk_counter = 0

        for page_info in pages:
            page_num = page_info.get("page", 1)
            page_text = page_info.get("text", "")

            text_pieces = self.split_text(page_text)
            for piece in text_pieces:
                chunk_id = f"{doc_id}_p{page_num}_c{chunk_counter}"
                all_chunks.append({
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "document": doc_name,
                    "career": career,
                    "page": page_num,
                    "chunk_index": chunk_counter,
                    "text": piece,
                    "char_count": len(piece)
                })
                chunk_counter += 1

        return all_chunks
