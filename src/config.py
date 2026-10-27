"""
Módulo de Configuración Centralizada.

Carga variables de entorno, define constantes de la institución educativa,
establece las carreras disponibles y gestiona las rutas de almacenamiento.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from src.utils.paths import get_bundle_dir, get_app_data_dir

# Cargar variables de entorno desde .env si existe en la raíz del proyecto
env_path = get_bundle_dir() / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()  # Intenta cargar de variables del sistema

class Config:
    """Configuración principal de la aplicación."""
    
    APP_NAME = "Chatbot Académico RAG"
    APP_VERSION = "0.5.0 - Etapa 5"
    
    # 3 Carreras de la institución para organizar los apuntes
    CAREERS = [
        "Ciberseguridad",
        "Ciencia de datos",
        "Desarrollo de software",
    ]
    ALL_CAREERS_LABEL = "Todas las carreras"
    
    # Rutas de almacenamiento
    DATA_DIR = get_app_data_dir()
    UPLOADS_DIR = DATA_DIR / "uploads"
    VECTORDB_DIR = DATA_DIR / "vectordb"
    
    # Configuración de Gemini API (Nivel Gratuito)
    # Límite gratuito típico: 15 RPM (peticiones por minuto), 1,500 RPD (peticiones por día)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    @classmethod
    def initialize_directories(cls) -> None:
        """Crea las carpetas necesarias para almacenar datos si no existen."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
        
        # Subcarpetas por carrera para organizar los archivos importados
        for career in cls.CAREERS:
            career_folder = cls.UPLOADS_DIR / career.replace(" ", "_").lower()
            career_folder.mkdir(parents=True, exist_ok=True)

    @classmethod
    def set_api_key(cls, key: str) -> None:
        """Actualiza la clave en memoria y en el archivo .env local para persistirla."""
        cls.GEMINI_API_KEY = key.strip()
        env_file = get_bundle_dir() / ".env"
        try:
            with open(env_file, "w", encoding="utf-8") as f:
                f.write(f"GEMINI_API_KEY={cls.GEMINI_API_KEY}\n")
                f.write(f"GEMINI_MODEL={cls.GEMINI_MODEL}\n")
        except Exception as e:
            print(f"[Config] Advertencia: No se pudo guardar la clave en .env: {e}")

    @classmethod
    def has_api_key(cls) -> bool:
        """Verifica si hay una clave API configurada."""
        return bool(cls.GEMINI_API_KEY)
