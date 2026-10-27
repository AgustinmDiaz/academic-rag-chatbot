"""
Módulo de gestión de rutas multiplataforma.

Garantiza compatibilidad entre macOS (Apple Silicon/Intel) y Windows,
así como compatibilidad con ejecutables congelados con PyInstaller.
"""

import os
import sys
import platform
from pathlib import Path


def is_frozen() -> bool:
    """Indica si la aplicación se está ejecutando desde un binario congelado (PyInstaller)."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_bundle_dir() -> Path:
    """
    Obtiene el directorio base de recursos de la aplicación.
    - En desarrollo: la raíz del proyecto.
    - En PyInstaller: el directorio temporal _MEIPASS o la carpeta del ejecutable.
    """
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent.parent


def get_app_data_dir(app_name: str = "AcademicRAGChatbot") -> Path:
    """
    Retorna la ruta estándar para almacenar datos persistentes del usuario
    (documentos indexados, base vectorial, configuración local),
    siguiendo las convenciones de cada sistema operativo:

    - macOS:   ~/Library/Application Support/<app_name>
    - Windows: %APPDATA%\\<app_name>  (habitualmente C:\\Users\\<usuario>\\AppData\\Roaming\\<app_name>)
    - Linux:   ~/.local/share/<app_name> (estándar XDG)

    En modo desarrollo, si existe una carpeta './data' en el proyecto,
    prioriza esa carpeta local para facilitar la inspección y depuración.
    """
    # En desarrollo usamos la carpeta data/ local para que el estudiante
    # pueda ver y revisar fácilmente los archivos generados.
    dev_data = get_bundle_dir() / "data"
    if not is_frozen() and dev_data.exists():
        return dev_data

    system = platform.system()
    home = Path.home()

    if system == "Darwin":  # macOS
        data_dir = home / "Library" / "Application Support" / app_name
    elif system == "Windows":  # Windows
        appdata = os.environ.get("APPDATA")
        if appdata:
            data_dir = Path(appdata) / app_name
        else:
            data_dir = home / "AppData" / "Roaming" / app_name
    else:  # Linux / Unix
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        if xdg_data_home:
            data_dir = Path(xdg_data_home) / app_name
        else:
            data_dir = home / ".local" / "share" / app_name

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
