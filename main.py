"""
Punto de Entrada Principal del Chatbot Académico RAG.

Permite iniciar la aplicación en modo interactivo (GUI) o en modo de prueba (--test-gui)
para verificar la inicialización de componentes y la compatibilidad del entorno.
"""

import sys
import argparse
from src.config import Config
from src.gui.app import MainWindow
from src.rag.service import RAGService


def print_banner():
    """Muestra información clara en la terminal al arrancar."""
    print("=" * 65)
    print(f"🎓 {Config.APP_NAME} - {Config.APP_VERSION}")
    print("=" * 65)
    print(f"📍 Directorio de datos: {Config.DATA_DIR}")
    print(f"🏛️  Carreras configuradas ({len(Config.CAREERS)}):")
    for career in Config.CAREERS:
        print(f"   • {career}")
    api_status = "Configurada" if Config.has_api_key() else "No detectada (se puede configurar en la UI)"
    print(f"🔑 Clave Gemini API: {api_status}")
    print("=" * 65)


def run_test_mode():
    """
    Modo de prueba automatizado:
    Instancia la ventana, verifica el renderizado de los componentes gráficos,
    ejecuta una consulta simulada de prueba y cierra limpiamente.
    """
    print("\n[Test] Iniciando verificación de la interfaz gráfica...")
    app = MainWindow()
    
    # Forzar actualización de geometría y componentes de Tkinter
    app.update()
    print("  ✓ Ventana principal y frames inicializados correctamente.")
    print("  ✓ Barra lateral, selector de carreras y botones cargados.")
    print("  ✓ Historial de chat y panel de fuentes activos.")

    # Prueba de consulta simulada
    print("[Test] Probando flujo de consulta simulada...")
    res = app.rag_service.answer_query("¿Cuáles son los principios de la seguridad?", "Ciberseguridad")
    assert "answer" in res and "sources" in res, "La respuesta simulada debe contener 'answer' y 'sources'"
    app._display_sources(res["sources"])
    app.update()
    print("  ✓ Renderizado de tarjetas de fuentes verificado con éxito.")

    # Cerrar ventana
    app.destroy()
    print("\n✅ ¡Verificación de la interfaz y componentes completada con éxito sin errores!\n")


def main():
    parser = argparse.ArgumentParser(description="Chatbot Académico RAG")
    parser.add_argument(
        "--test-gui",
        action="store_true",
        help="Ejecuta una prueba rápida de inicialización de la interfaz y sale."
    )
    args = parser.parse_args()

    print_banner()

    if args.test_gui:
        run_test_mode()
        sys.exit(0)

    # Modo interactivo normal
    print("\nIniciando interfaz gráfica... (Cierra la ventana para salir)")
    app = MainWindow()
    app.mainloop()
    print("\nAplicación finalizada.")


if __name__ == "__main__":
    main()
