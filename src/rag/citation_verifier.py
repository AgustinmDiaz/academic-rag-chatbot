"""
Módulo de Verificación de Citas y Consistencia Documental.

Garantiza el principio de CERO ALUCINACIONES comprobando que:
1. Las afirmaciones de la respuesta citen documentos y páginas válidas.
2. Cada cita emitida por el modelo corresponda a un fragmento realmente recuperado en ChromaDB.
3. Se detecten citas inventadas o páginas inexistentes en el contexto.
"""

import re
from typing import List, Dict, Any, Tuple


class CitationVerifier:
    """Verificador de fidelidad de citas y contraste con el contexto recuperado."""

    # Expresión regular para detectar citas en formatos comunes:
    # [Documento: X, Pág. Y], [Doc: X, Pag. Y], [X, Pág. Y], [Pág. Y - X]
    CITATION_PATTERNS = [
        re.compile(r"\[(?:Documento:\s*|Doc:\s*)?([^,\]\n]+?),\s*(?:Pág\.?|Pag\.?|Página)\s*(\d+)\]", re.IGNORECASE),
        re.compile(r"\[(?:Pág\.?|Pag\.?|Página)\s*(\d+)\s*(?:-|de|en)\s*([^,\]\n]+?)\]", re.IGNORECASE),
    ]

    @classmethod
    def extract_citations(cls, text: str) -> List[Dict[str, Any]]:
        """Extrae todas las referencias de citas encontradas en el texto de respuesta."""
        citations = []
        seen = set()

        # Patrón 1: [Documento: X, Pág. Y]
        for match in cls.CITATION_PATTERNS[0].finditer(text):
            doc_name = match.group(1).strip()
            page_num = int(match.group(2).strip())
            key = (doc_name.lower(), page_num)
            if key not in seen:
                seen.add(key)
                citations.append({
                    "document": doc_name,
                    "page": page_num,
                    "raw_citation": match.group(0)
                })

        # Patrón 2: [Pág. Y - X]
        for match in cls.CITATION_PATTERNS[1].finditer(text):
            page_num = int(match.group(1).strip())
            doc_name = match.group(2).strip()
            key = (doc_name.lower(), page_num)
            if key not in seen:
                seen.add(key)
                citations.append({
                    "document": doc_name,
                    "page": page_num,
                    "raw_citation": match.group(0)
                })

        return citations

    @classmethod
    def verify_citations(
        cls,
        answer_text: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verifica que todas las citas en la respuesta del modelo tengan sustento
        en los fragmentos recuperados en ChromaDB.
        """
        extracted = cls.extract_citations(answer_text)

        # Crear conjunto de pares válidos (documento_limpio, pagina) en el contexto recuperado
        valid_pairs = set()
        for chunk in retrieved_chunks:
            doc = chunk.get("document", "").strip().lower()
            page = int(chunk.get("page", 1))
            valid_pairs.add((doc, page))

        valid_citations = []
        hallucinated_citations = []

        for cite in extracted:
            cite_doc = cite["document"].strip().lower()
            cite_page = cite["page"]

            # Comprobar si el documento citado coincide total o parcialmente con alguno recuperado
            match_found = False
            for (valid_doc, valid_page) in valid_pairs:
                if (cite_doc in valid_doc or valid_doc in cite_doc) and cite_page == valid_page:
                    match_found = True
                    break

            if match_found:
                valid_citations.append(cite)
            else:
                hallucinated_citations.append(cite)

        is_fully_grounded = len(hallucinated_citations) == 0
        fidelity_score = (
            len(valid_citations) / len(extracted) if extracted else (1.0 if not retrieved_chunks else 0.0)
        )

        return {
            "is_fully_grounded": is_fully_grounded,
            "total_citations": len(extracted),
            "valid_citations": valid_citations,
            "hallucinated_citations": hallucinated_citations,
            "fidelity_score": round(fidelity_score, 2),
            "has_citations": len(extracted) > 0
        }
