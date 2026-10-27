"""
Módulo de Extracción de Texto para Documentos Académicos (PDF, TXT e Imágenes OCR).

Extrae texto página por página preservando la trazabilidad de citas:
- PDF: Extracción página a página con pypdf.
- TXT: Extracción con detección de codificación (UTF-8, Latin-1, CP1252).
- Imágenes (PNG, JPG, JPEG, WEBP): Extracción mediante OCR (Tesseract local o Gemini Vision gratuito).
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import shutil

from src.utils.paths import get_bundle_dir
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from PIL import Image
import pytesseract

from src.config import Config

try:
    from google import genai
    from google.genai import types
    GENAI_VISION_AVAILABLE = True
except ImportError:
    GENAI_VISION_AVAILABLE = False


class DocumentExtractionError(Exception):
    """Excepción base para errores de extracción de documentos."""
    pass


class EncryptedPDFError(DocumentExtractionError):
    """El archivo PDF está protegido con contraseña o encriptado."""
    pass


class EmptyDocumentError(DocumentExtractionError):
    """El archivo no contiene texto legible (puede estar vacío o ser un escaneo ilegible)."""
    pass


class OCREngineNotFoundError(DocumentExtractionError):
    """No se encontró un motor de OCR local (Tesseract) ni clave de Gemini para OCR multimodal."""
    pass


class DocumentExtractor:
    """Extrae texto de archivos PDF, TXT e imágenes con OCR."""

    TEXT_EXTENSIONS = {".txt"}
    PDF_EXTENSIONS = {".pdf"}
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | IMAGE_EXTENSIONS

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Limpia caracteres de control y normaliza espacios en blanco."""
        if not text:
            return ""
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _configure_tesseract(cls) -> None:
        """Busca primero el motor OCR incluido en la aplicación."""
        bundle_dir = get_bundle_dir()

        candidates = (
            bundle_dir / "tesseract" / "tesseract.exe",
            bundle_dir / "tesseract" / "tesseract",
            bundle_dir / "tesseract.exe",
            bundle_dir / "tesseract",
        )

        for candidate in candidates:
            if candidate.is_file():
                pytesseract.pytesseract.tesseract_cmd = str(candidate)
                return

        system_tesseract = shutil.which("tesseract")
        if system_tesseract:
            pytesseract.pytesseract.tesseract_cmd = system_tesseract

    @classmethod
    def _tesseract_config(cls) -> str:
        """Usa los datos de idioma que acompañan al motor OCR empaquetado."""
        cls._configure_tesseract()

        executable = Path(pytesseract.pytesseract.tesseract_cmd)
        tessdata_dir = executable.parent / "tessdata"

        if tessdata_dir.is_dir():
            return f'--tessdata-dir "{tessdata_dir}"'

        return ""

    @classmethod
    def is_tesseract_available(cls) -> bool:
        """Comprueba si el motor OCR configurado puede ejecutarse."""
        cls._configure_tesseract()

        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @classmethod
    def extract_txt(cls, file_path: Path) -> List[Dict[str, Any]]:
        """Extrae texto de un archivo de texto plano (TXT)."""
        content = ""
        encodings_to_try = ["utf-8", "latin-1", "cp1252"]
        success = False

        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                success = True
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if not success:
            raise DocumentExtractionError(f"No se pudo decodificar '{file_path.name}'.")

        cleaned = cls.clean_text(content)
        if not cleaned:
            raise EmptyDocumentError(f"El archivo '{file_path.name}' está vacío.")

        return [{
            "page": 1,
            "text": cleaned,
            "char_count": len(cleaned)
        }]

    @classmethod
    def extract_pdf(cls, file_path: Path) -> List[Dict[str, Any]]:
        """Extrae texto página por página de un PDF con pypdf."""
        try:
            reader = PdfReader(str(file_path))
        except PdfReadError as e:
            raise DocumentExtractionError(f"El PDF '{file_path.name}' está dañado: {e}")
        except Exception as e:
            raise DocumentExtractionError(f"Error al abrir el PDF '{file_path.name}': {e}")

        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
                if not decrypted:
                    raise EncryptedPDFError(f"El PDF '{file_path.name}' tiene contraseña.")
            except Exception:
                raise EncryptedPDFError(f"El PDF '{file_path.name}' tiene contraseña.")

        pages_data: List[Dict[str, Any]] = []
        total_text_length = 0

        for idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            cleaned = cls.clean_text(page_text)
            total_text_length += len(cleaned)

            pages_data.append({
                "page": idx,
                "text": cleaned,
                "char_count": len(cleaned)
            })

        if total_text_length == 0:
            raise EmptyDocumentError(
                f"El archivo PDF '{file_path.name}' no contiene texto seleccionable. "
                f"Es un escaneo o imagen pura sin capa de texto OCR."
            )

        return pages_data

    @classmethod
    def extract_image(cls, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extrae texto de una imagen (.png, .jpg, .jpeg) mediante OCR:
        1. Si Tesseract OCR está instalado, realiza OCR local 100% desconectado.
        2. Si Tesseract no está instalado pero hay clave de Gemini API configurada, utiliza Gemini Vision gratuito.
        3. Si ninguno está disponible, informa al usuario con instrucciones exactas.
        """
        raw_text = ""

        # Intento 1: Tesseract OCR local
        if cls.is_tesseract_available():
            try:
                with Image.open(file_path) as img:
                    raw_text = pytesseract.image_to_string(img, lang="spa+eng", config=cls._tesseract_config())
            except Exception as e:
                print(f"[DocumentExtractor] Advertencia en Tesseract local: {e}")
                raw_text = ""

        # Intento 2: Gemini Vision gratuito si Tesseract no está instalado o no extrajo texto
        if not raw_text.strip() and Config.has_api_key() and GENAI_VISION_AVAILABLE:
            try:
                client = genai.Client(api_key=Config.GEMINI_API_KEY)
                with open(file_path, "rb") as f:
                    image_bytes = f.read()

                suffix = file_path.suffix.lower().replace(".", "")
                mime_type = "image/png" if suffix == "png" else "image/jpeg"

                response = client.models.generate_content(
                    model=Config.GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        "Extrae todo el texto legible de esta imagen o apunte académico de forma exacta. "
                        "No agregues comentarios ni interpretaciones, solo devuelve el texto transcripto."
                    ]
                )
                raw_text = response.text or ""
            except Exception as e:
                print(f"[DocumentExtractor] Advertencia en Gemini Vision OCR: {e}")

        # Intento 3: Si no se pudo ejecutar ningún motor de OCR
        if not cls.is_tesseract_available() and not Config.has_api_key():
            raise OCREngineNotFoundError(
                f"No se pudo procesar la imagen '{file_path.name}'.\n\n"
                f"Para extraer texto de imágenes se requiere una de estas opciones gratuitas:\n"
                f"1. Instalar Tesseract OCR en tu sistema:\n"
                f"   • En macOS: ejecuta 'brew install tesseract' en la terminal.\n"
                f"   • En Windows: instala Tesseract desde el instalador oficial de GitHub.\n"
                f"2. O configurar tu clave gratuita de Google Gemini en la barra lateral ('🔑 Configurar Clave API'), "
                f"la cual incluye OCR multimodal automático."
            )

        cleaned = cls.clean_text(raw_text)
        if not cleaned:
            raise EmptyDocumentError(
                f"La imagen '{file_path.name}' no contiene texto legible detectable por OCR. "
                f"Asegúrate de que la imagen tenga buena resolución e iluminación."
            )

        return [{
            "page": 1,
            "text": cleaned,
            "char_count": len(cleaned)
        }]

    @classmethod
    def extract(cls, file_path: Path) -> List[Dict[str, Any]]:
        """Punto de entrada principal para extraer texto de PDF, TXT o imágenes."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"El archivo no existe: {file_path}")

        ext = path.suffix.lower()
        if ext in cls.TEXT_EXTENSIONS:
            return cls.extract_txt(path)
        elif ext in cls.PDF_EXTENSIONS:
            return cls.extract_pdf(path)
        elif ext in cls.IMAGE_EXTENSIONS:
            return cls.extract_image(path)
        else:
            raise ValueError(
                f"Extensión '{ext}' no soportada. Formatos compatibles: "
                f"PDF (.pdf), TXT (.txt) e Imágenes (.png, .jpg, .jpeg, .webp, .bmp)."
            )
