"""
Módulo de Interfaz Gráfica de Usuario (GUI) con CustomTkinter.

Implementa la ventana principal del Chatbot Académico RAG con:
- Barra lateral: selector de carreras, administración documental y estado de API.
- Panel central: historial de chat enriquecido con burbujas y notas de citas verificadas.
- Panel lateral derecho: visor interactivo de fuentes, páginas, porcentajes de similitud y copiado de citas.
- Barra inferior: estado del sistema e indicadores de verificación de fuentes.
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.config import Config
from src.rag.service import RAGService
from src.rag.document_manager import DuplicateDocumentError
from src.rag.extractor import (
    DocumentExtractionError,
    EmptyDocumentError,
    EncryptedPDFError,
    OCREngineNotFoundError
)


# Configuración estética global
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class MainWindow(ctk.CTk):
    """Ventana principal de la aplicación."""

    def __init__(self, rag_service: Optional[RAGService] = None):
        super().__init__()

        self.rag_service = rag_service or RAGService()
        Config.initialize_directories()

        # Configuración básica de la ventana
        self.title(f"{Config.APP_NAME} - {Config.APP_VERSION}")
        self.geometry("1120x740")
        self.minsize(940, 620)

        # Configuración del grid principal (3 columnas)
        self.grid_columnconfigure(0, weight=0)  # Sidebar
        self.grid_columnconfigure(1, weight=1)  # Chat central
        self.grid_columnconfigure(2, weight=0)  # Fuentes
        self.grid_rowconfigure(0, weight=1)     # Contenido principal
        self.grid_rowconfigure(1, weight=0)     # Barra de estado inferior

        # Variables reactivas de la UI
        self.current_career_var = ctk.StringVar(value=Config.ALL_CAREERS_LABEL)
        self.status_var = ctk.StringVar(value="Sistema listo para recibir consultas.")
        self.sources_visible = True

        # Construcción de componentes
        self._build_sidebar()
        self._build_chat_panel()
        self._build_sources_panel()
        self._build_status_bar()

        # Cargar lista inicial de documentos
        self._refresh_documents_list()

    # =========================================================================
    # BARRA LATERAL (Sidebar): Carreras, Documentos y Configuración
    # =========================================================================
    def _build_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=290, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # 1. Título y logo institucional
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="🎓 Campus Virtual\nChatbot RAG",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 15), sticky="w")

        # 2. Selector de Carrera
        self.career_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Filtrar por Carrera:",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.career_label.grid(row=1, column=0, padx=20, pady=(5, 2), sticky="w")

        career_options = [Config.ALL_CAREERS_LABEL] + Config.CAREERS
        self.career_menu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=career_options,
            variable=self.current_career_var,
            command=self._on_career_changed
        )
        self.career_menu.grid(row=2, column=0, padx=20, pady=(0, 15), sticky="ew")

        # 3. Encabezado de Documentos con contador
        self.docs_header_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.docs_header_frame.grid(row=3, column=0, padx=20, pady=(5, 2), sticky="ew")
        self.docs_header_frame.grid_columnconfigure(0, weight=1)

        self.docs_header = ctk.CTkLabel(
            self.docs_header_frame,
            text="📚 Materiales Cargados:",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.docs_header.grid(row=0, column=0, sticky="w")

        self.docs_count_label = ctk.CTkLabel(
            self.docs_header_frame,
            text="(0)",
            font=ctk.CTkFont(size=12),
            text_color="#888888"
        )
        self.docs_count_label.grid(row=0, column=1, sticky="e")

        # 4. Lista de Documentos con Scroll
        self.docs_list_frame = ctk.CTkScrollableFrame(self.sidebar_frame, height=220)
        self.docs_list_frame.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="nsew")

        # 5. Botones de Gestión de Documentos
        self.docs_actions_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.docs_actions_frame.grid(row=5, column=0, padx=20, pady=(0, 15), sticky="ew")
        self.docs_actions_frame.grid_columnconfigure((0, 1), weight=1)

        self.import_btn = ctk.CTkButton(
            self.docs_actions_frame,
            text="📥 Importar",
            command=self._on_import_clicked,
            fg_color="#1f538d",
            hover_color="#14375e"
        )
        self.import_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.clear_btn = ctk.CTkButton(
            self.docs_actions_frame,
            text="🗑️ Vaciar Todo",
            command=self._on_clear_clicked,
            fg_color="#6c757d",
            hover_color="#495057"
        )
        self.clear_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        # 6. Estado de la API de Gemini
        self.api_frame = ctk.CTkFrame(self.sidebar_frame)
        self.api_frame.grid(row=6, column=0, padx=20, pady=(0, 15), sticky="ew")

        has_key = Config.has_api_key()
        api_text = "🟢 API Gemini: Configurada" if has_key else "🟡 API Gemini: Sin clave"
        self.api_status_label = ctk.CTkLabel(
            self.api_frame,
            text=api_text,
            font=ctk.CTkFont(size=11)
        )
        self.api_status_label.pack(padx=10, pady=(8, 4))

        self.api_key_btn = ctk.CTkButton(
            self.api_frame,
            text="🔑 Configurar Clave API",
            command=self._open_api_key_dialog,
            height=26,
            font=ctk.CTkFont(size=11)
        )
        self.api_key_btn.pack(padx=10, pady=(0, 8), fill="x")

        # 7. Selector de Tema (Claro / Oscuro)
        self.theme_menu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["System", "Dark", "Light"],
            command=ctk.set_appearance_mode
        )
        self.theme_menu.grid(row=7, column=0, padx=20, pady=(0, 15), sticky="ew")

    # =========================================================================
    # PANEL CENTRAL: Historial de Chat y Entrada de Preguntas
    # =========================================================================
    def _build_chat_panel(self):
        self.chat_container = ctk.CTkFrame(self)
        self.chat_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.chat_container.grid_rowconfigure(1, weight=1)
        self.chat_container.grid_columnconfigure(0, weight=1)

        # 1. Cabecera del Chat
        self.chat_header = ctk.CTkFrame(self.chat_container, fg_color="transparent")
        self.chat_header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))
        self.chat_header.grid_columnconfigure(0, weight=1)

        self.chat_title = ctk.CTkLabel(
            self.chat_header,
            text="💬 Consultas Académicas",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.chat_title.grid(row=0, column=0, sticky="w")

        self.toggle_sources_btn = ctk.CTkButton(
            self.chat_header,
            text="📚 Ocultar Fuentes",
            width=120,
            height=26,
            command=self._toggle_sources_panel
        )
        self.toggle_sources_btn.grid(row=0, column=1, sticky="e")

        # 2. Historial de Mensajes
        self.chat_history_frame = ctk.CTkScrollableFrame(self.chat_container)
        self.chat_history_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=5)
        self.chat_history_frame.grid_columnconfigure(0, weight=1)

        # Mensaje de bienvenida inicial
        self._add_chat_bubble(
            sender="Asistente Académico",
            message=(
                "¡Hola! Soy tu asistente de estudio institucional con tecnología RAG.\n\n"
                "Para comenzar, importa los apuntes (PDF o TXT) de tu carrera usando la barra lateral.\n\n"
                "🛡️ Principio de Cero Alucinaciones: Todas las respuestas son verificadas contra los materiales oficiales y respaldadas con citas exactas."
            ),
            is_user=False
        )

        # 3. Indicador de procesamiento
        self.typing_label = ctk.CTkLabel(
            self.chat_container,
            text="⏳ Analizando documentos, recuperando vectores y redactando...",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="#888888"
        )

        # 4. Campo de Entrada y Envío
        self.input_frame = ctk.CTkFrame(self.chat_container, fg_color="transparent")
        self.input_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(5, 15))
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.query_entry = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="Escribe tu consulta académica aquí...",
            height=40,
            font=ctk.CTkFont(size=13)
        )
        self.query_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.query_entry.bind("<Return>", lambda event: self._send_query())

        self.send_btn = ctk.CTkButton(
            self.input_frame,
            text="Enviar 📤",
            width=100,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._send_query
        )
        self.send_btn.grid(row=0, column=1, sticky="e")

    # =========================================================================
    # PANEL DERECHO: Visualizador Refinado de Fuentes y Citas (ETAPA 5)
    # =========================================================================
    def _build_sources_panel(self):
        self.sources_frame = ctk.CTkFrame(self, width=300)
        self.sources_frame.grid(row=0, column=2, sticky="nsew", padx=(0, 10), pady=10)
        self.sources_frame.grid_rowconfigure(2, weight=1)

        # Encabezado del panel
        self.sources_title = ctk.CTkLabel(
            self.sources_frame,
            text="📖 Fuentes y Citas",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.sources_title.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        # Badge de verificación de citas
        self.verification_badge = ctk.CTkLabel(
            self.sources_frame,
            text="🛡️ Cero Alucinaciones Activo",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#17a2b8"
        )
        self.verification_badge.grid(row=1, column=0, padx=15, pady=(0, 8), sticky="w")

        # Contenedor de tarjetas
        self.sources_content = ctk.CTkScrollableFrame(self.sources_frame, width=280)
        self.sources_content.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")

        self.no_sources_label = ctk.CTkLabel(
            self.sources_content,
            text="Las fuentes consultadas, páginas citadas y porcentajes de similitud se mostrarán aquí luego de cada consulta.",
            wraplength=240,
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="#888888"
        )
        self.no_sources_label.pack(padx=10, pady=25)

    # =========================================================================
    # BARRA DE ESTADO INFERIOR
    # =========================================================================
    def _build_status_bar(self):
        self.status_bar = ctk.CTkFrame(self, height=28, corner_radius=0)
        self.status_bar.grid(row=1, column=0, columnspan=3, sticky="ew")

        self.status_label = ctk.CTkLabel(
            self.status_bar,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        self.status_label.pack(side="left", padx=15, pady=2)

        self.version_label = ctk.CTkLabel(
            self.status_bar,
            text=f"{Config.APP_NAME} | RAG Cero Alucinaciones",
            font=ctk.CTkFont(size=11),
            anchor="e"
        )
        self.version_label.pack(side="right", padx=15, pady=2)

    # =========================================================================
    # VISUALIZACIÓN REFINADA DE FUENTES Y CITAS (ETAPA 5)
    # =========================================================================
    def _display_sources(self, sources: List[Dict[str, Any]], verification: Optional[Dict[str, Any]] = None):
        """Muestra tarjetas enriquecidas con documentos, páginas, porcentajes y verificación."""
        for widget in self.sources_content.winfo_children():
            widget.destroy()

        if not sources:
            self.verification_badge.configure(
                text="ℹ️ Sin citas externas requeridas",
                text_color="#888888"
            )
            label = ctk.CTkLabel(
                self.sources_content,
                text="No se requirieron fuentes para esta respuesta.",
                wraplength=240,
                font=ctk.CTkFont(size=12, slant="italic")
            )
            label.pack(padx=10, pady=25)
            return

        # Actualizar badge de verificación
        if verification and verification.get("is_fully_grounded", True) and verification.get("has_citations", False):
            self.verification_badge.configure(
                text=f"✅ {verification.get('total_citations', 0)} Cita(s) 100% Verificada(s)",
                text_color="#28a745"
            )
        elif verification and not verification.get("is_fully_grounded", True):
            self.verification_badge.configure(
                text="⚠️ Alerta: Cita no contrastada en apuntes",
                text_color="#ffc107"
            )
        else:
            self.verification_badge.configure(
                text=f"📚 {len(sources)} Fragmento(s) Recuperado(s)",
                text_color="#17a2b8"
            )

        for idx, src in enumerate(sources, 1):
            card = ctk.CTkFrame(self.sources_content, corner_radius=8, fg_color=("gray90", "gray20"))
            card.pack(fill="x", padx=4, pady=6)

            # Fila 1: Título del documento y página
            doc_name = src.get("document", "Documento")
            page_num = src.get("page", 1)

            header_frame = ctk.CTkFrame(card, fg_color="transparent")
            header_frame.pack(fill="x", padx=8, pady=(8, 2))
            header_frame.grid_columnconfigure(0, weight=1)

            title_label = ctk.CTkLabel(
                header_frame,
                text=f"📄 #{idx}: {doc_name[:18]}..." if len(doc_name) > 20 else f"📄 #{idx}: {doc_name}",
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor="w"
            )
            title_label.grid(row=0, column=0, sticky="w")

            page_badge = ctk.CTkLabel(
                header_frame,
                text=f"Pág. {page_num}",
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color="#1f538d",
                text_color="#ffffff",
                corner_radius=4,
                width=55,
                height=18
            )
            page_badge.grid(row=0, column=1, sticky="e")

            # Fila 2: Carrera y porcentaje de similitud
            sim = src.get("similarity", 0.0)
            sim_percent = int(sim * 100)
            sim_color = "#28a745" if sim >= 0.70 else ("#17a2b8" if sim >= 0.50 else "#6c757d")

            meta_frame = ctk.CTkFrame(card, fg_color="transparent")
            meta_frame.pack(fill="x", padx=8, pady=(0, 4))
            meta_frame.grid_columnconfigure(0, weight=1)

            career_lbl = ctk.CTkLabel(
                meta_frame,
                text=src.get("career", ""),
                font=ctk.CTkFont(size=9),
                text_color="#888888",
                anchor="w"
            )
            career_lbl.grid(row=0, column=0, sticky="w")

            sim_lbl = ctk.CTkLabel(
                meta_frame,
                text=f"Relevancia: {sim_percent}%",
                font=ctk.CTkFont(size=9, weight="bold"),
                text_color=sim_color,
                anchor="e"
            )
            sim_lbl.grid(row=0, column=1, sticky="e")

            # Fila 3: Fragmento textual citado
            snippet = src.get("fragment", "")
            snippet_box = ctk.CTkLabel(
                card,
                text=f'"{snippet}"',
                wraplength=235,
                justify="left",
                font=ctk.CTkFont(size=11, slant="italic"),
                anchor="w"
            )
            snippet_box.pack(fill="x", padx=8, pady=(0, 6))

            # Fila 4: Botón para copiar fragmento
            btn_frame = ctk.CTkFrame(card, fg_color="transparent")
            btn_frame.pack(fill="x", padx=8, pady=(0, 8))

            copy_btn = ctk.CTkButton(
                btn_frame,
                text="📋 Copiar cita",
                height=20,
                width=90,
                font=ctk.CTkFont(size=10),
                fg_color=("gray75", "gray30"),
                text_color=("black", "white"),
                command=lambda t=snippet: self._copy_to_clipboard(t)
            )
            copy_btn.pack(side="right")

    def _copy_to_clipboard(self, text: str):
        """Copia el texto del fragmento al portapapeles del sistema operativo."""
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set("Cita copiada al portapapeles.")

    # =========================================================================
    # LÓGICA DE CHAT Y COMUNICACIÓN ASÍNCRONA
    # =========================================================================
    def _add_chat_bubble(self, sender: str, message: str, is_user: bool, verification_note: Optional[str] = None):
        """Agrega un mensaje con estilo diferenciado y pie de cita verificada."""
        bubble = ctk.CTkFrame(
            self.chat_history_frame,
            fg_color="#1f538d" if is_user else ("#e9ecef", "#2b2b2b"),
            corner_radius=12
        )
        bubble.pack(fill="x", padx=10, pady=6, anchor="e" if is_user else "w")

        header = ctk.CTkLabel(
            bubble,
            text=sender,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#ffffff" if is_user else ("#1a1a1a", "#e0e0e0")
        )
        header.pack(anchor="w", padx=12, pady=(8, 2))

        msg_body = ctk.CTkLabel(
            bubble,
            text=message,
            wraplength=520,
            justify="left",
            font=ctk.CTkFont(size=13),
            text_color="#ffffff" if is_user else ("#212529", "#f8f9fa")
        )
        msg_body.pack(anchor="w", padx=12, pady=(0, 8))

        # Nota de verificación de citas al pie del mensaje
        if verification_note and not is_user:
            footer = ctk.CTkLabel(
                bubble,
                text=verification_note,
                font=ctk.CTkFont(size=10, slant="italic"),
                text_color="#17a2b8" if "100%" in verification_note else "#ffc107"
            )
            footer.pack(anchor="w", padx=12, pady=(0, 6))

        self.after(100, lambda: self.chat_history_frame._parent_canvas.yview_moveto(1.0))

    def _send_query(self):
        """Envía la consulta del usuario, actualiza la UI y ejecuta en segundo plano."""
        query = self.query_entry.get().strip()
        if not query:
            return

        career = self.current_career_var.get()

        self._add_chat_bubble(sender="Estudiante", message=query, is_user=True)
        self.query_entry.delete(0, tk.END)

        self.send_btn.configure(state="disabled")
        self.typing_label.grid(row=2, column=0, sticky="w", padx=20, pady=2)
        self.status_var.set(f"Recuperando fragmentos y verificando citas ({career})...")

        threading.Thread(
            target=self._process_query_thread,
            args=(query, career),
            daemon=True
        ).start()

    def _process_query_thread(self, query: str, career: str):
        """Ejecuta la búsqueda RAG y agenda la actualización de la UI en el hilo principal."""
        try:
            result = self.rag_service.answer_query(query, career_filter=career)
            self.after(0, self._on_query_completed, result)
        except Exception as e:
            self.after(0, self._on_query_error, str(e))

    def _on_query_completed(self, result: Dict[str, Any]):
        """Callback cuando la respuesta está lista."""
        self.typing_label.grid_forget()
        self.send_btn.configure(state="normal")
        self.status_var.set("Respuesta generada y citas verificadas con éxito.")

        verification = result.get("verification", {})
        note = None
        if verification.get("has_citations"):
            note = f"🛡️ Citas verificadas en los apuntes: {verification.get('total_citations', 0)} referencia(s)"

        self._add_chat_bubble(
            sender="Asistente Académico",
            message=result.get("answer", "No se obtuvo respuesta."),
            is_user=False,
            verification_note=note
        )
        self._display_sources(result.get("sources", []), verification=verification)

    def _on_query_error(self, error_message: str):
        """Callback ante errores durante la consulta."""
        self.typing_label.grid_forget()
        self.send_btn.configure(state="normal")
        self.status_var.set("Error al procesar la consulta.")
        self._add_chat_bubble(
            sender="Sistema",
            message=f"❌ Error al procesar tu consulta:\n{error_message}",
            is_user=False
        )

    # =========================================================================
    # GESTIÓN DE DOCUMENTOS Y EVENTOS GENERALES
    # =========================================================================
    def _refresh_documents_list(self):
        """Actualiza la lista visual de documentos en la barra lateral."""
        for widget in self.docs_list_frame.winfo_children():
            widget.destroy()

        career = self.current_career_var.get()
        docs = self.rag_service.list_documents(career_filter=career)
        self.docs_count_label.configure(text=f"({len(docs)})")

        if not docs:
            empty_lbl = ctk.CTkLabel(
                self.docs_list_frame,
                text="Sin documentos cargados.\nHaz clic en '📥 Importar'.",
                wraplength=210,
                font=ctk.CTkFont(size=11, slant="italic"),
                text_color="#888888"
            )
            empty_lbl.pack(padx=5, pady=25)
            return

        for doc in docs:
            row = ctk.CTkFrame(self.docs_list_frame, fg_color=("gray85", "gray22"), corner_radius=6)
            row.pack(fill="x", padx=2, pady=3)
            row.grid_columnconfigure(0, weight=1)

            name = doc.get("name", "Documento")
            ext = doc.get("extension", "").lower()
            icon = "🖼️" if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp") else "📄"
            display_name = f"{icon} {name[:18]}..." if len(name) > 20 else f"{icon} {name}"

            info_frame = ctk.CTkFrame(row, fg_color="transparent")
            info_frame.grid(row=0, column=0, sticky="w", padx=6, pady=4)

            name_lbl = ctk.CTkLabel(
                info_frame,
                text=display_name,
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor="w"
            )
            name_lbl.pack(anchor="w")

            pages = doc.get("pages_count", 1)
            p_text = "OCR" if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp") else (f"{pages} pág." if pages == 1 else f"{pages} págs.")
            meta_text = f"{p_text} • {doc.get('career', '')}"
            meta_lbl = ctk.CTkLabel(
                info_frame,
                text=meta_text,
                font=ctk.CTkFont(size=9),
                text_color="#888888",
                anchor="w"
            )
            meta_lbl.pack(anchor="w")

            doc_id = doc.get("id")
            del_btn = ctk.CTkButton(
                row,
                text="❌",
                width=24,
                height=24,
                fg_color="transparent",
                hover_color="#c82333",
                text_color="#ff5555",
                font=ctk.CTkFont(size=11),
                command=lambda d_id=doc_id, d_name=name: self._on_delete_single_doc(d_id, d_name)
            )
            del_btn.grid(row=0, column=1, sticky="e", padx=(0, 4))

    def _on_delete_single_doc(self, doc_id: str, doc_name: str):
        """Elimina un documento individual tras confirmar."""
        confirm = messagebox.askyesno(
            "Eliminar Documento",
            f"¿Deseas eliminar el apunte '{doc_name}' del sistema?"
        )
        if confirm:
            success = self.rag_service.delete_document(doc_id)
            if success:
                self.status_var.set(f"Se eliminó '{doc_name}'.")
                self._refresh_documents_list()
            else:
                messagebox.showerror("Error", f"No se pudo eliminar el documento '{doc_name}'.")

    def _on_import_clicked(self):
        """Maneja la importación real de archivos con deduplicación y manejo de errores."""
        file_paths = filedialog.askopenfilenames(
            title="Seleccionar apuntes académicos o imágenes con texto",
            filetypes=[
                ("Todos los compatibles (*.pdf, *.txt, *.png, *.jpg)", "*.pdf *.txt *.png *.jpg *.jpeg *.webp *.bmp"),
                ("Archivos PDF (*.pdf)", "*.pdf"),
                ("Archivos de texto (*.txt)", "*.txt"),
                ("Imágenes con texto OCR (*.png, *.jpg, *.jpeg)", "*.png *.jpg *.jpeg *.webp *.bmp"),
            ]
        )
        if not file_paths:
            return

        career = self.current_career_var.get()
        if career == Config.ALL_CAREERS_LABEL:
            career = Config.CAREERS[0]
            self.current_career_var.set(career)

        imported_count = 0
        duplicates = []
        errors = []

        for p_str in file_paths:
            file_path = Path(p_str)
            try:
                self.rag_service.add_document(file_path, career)
                imported_count += 1
            except DuplicateDocumentError:
                duplicates.append(file_path.name)
            except OCREngineNotFoundError as e:
                errors.append(f"{file_path.name}: {e}")
            except (EmptyDocumentError, EncryptedPDFError, DocumentExtractionError) as e:
                errors.append(f"{file_path.name}: {e}")
            except Exception as e:
                errors.append(f"{file_path.name}: Error inesperado: {e}")

        self._refresh_documents_list()

        summary_lines = []
        if imported_count > 0:
            summary_lines.append(f"✅ {imported_count} documento(s) indexado(s) en '{career}'.")
            self.status_var.set(f"Se importaron {imported_count} apunte(s) en {career}.")

        if duplicates:
            summary_lines.append(f"\n⚠️ Duplicados omitidos ({len(duplicates)}):")
            for dup in duplicates:
                summary_lines.append(f"  • {dup} (contenido idéntico ya cargado)")

        if errors:
            summary_lines.append(f"\n❌ Errores de lectura ({len(errors)}):")
            for err in errors:
                summary_lines.append(f"  • {err}")

        msg = "\n".join(summary_lines)
        if errors:
            messagebox.showwarning("Resultado de la Importación", msg)
        elif duplicates and imported_count == 0:
            messagebox.showinfo("Documentos Duplicados", msg)
        else:
            messagebox.showinfo("Importación Completada", msg)

    def _on_clear_clicked(self):
        """Vacía todos los documentos tras confirmar."""
        confirm = messagebox.askyesno(
            "Confirmar vaciado",
            "¿Estás seguro de que deseas eliminar TODOS los documentos indexados?\n"
            "Esta acción no se puede deshacer."
        )
        if confirm:
            deleted_count = self.rag_service.clear_documents()
            self._refresh_documents_list()
            self.status_var.set(f"Se eliminaron todos los documentos ({deleted_count}).")
            messagebox.showinfo("Documentos eliminados", f"Se eliminaron {deleted_count} documento(s).")

    def _on_career_changed(self, selected_career: str):
        """Maneja el cambio de filtro de carrera."""
        self.status_var.set(f"Filtro activo: {selected_career}")
        self._refresh_documents_list()

    def _toggle_sources_panel(self):
        """Oculta o muestra el panel lateral de fuentes."""
        if self.sources_visible:
            self.sources_frame.grid_remove()
            self.toggle_sources_btn.configure(text="📚 Mostrar Fuentes")
            self.sources_visible = False
        else:
            self.sources_frame.grid()
            self.toggle_sources_btn.configure(text="📚 Ocultar Fuentes")
            self.sources_visible = True

    def _open_api_key_dialog(self):
        """Abre un diálogo sencillo para ingresar la clave de Gemini sin editar código."""
        dialog = ctk.CTkInputDialog(
            text="Pega tu clave gratuita de Google Gemini API:\n(Se guardará localmente en .env y nunca se compartirá)",
            title="Configurar Gemini API Key"
        )
        key = dialog.get_input()
        if key is not None and key.strip():
            Config.set_api_key(key.strip())
            self.api_status_label.configure(text="🟢 API Gemini: Configurada")
            self.status_var.set("Clave API guardada correctamente.")
            messagebox.showinfo("Configuración", "Clave API configurada con éxito.")
