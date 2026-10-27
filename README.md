# 🎓 Chatbot Académico RAG (100% Gratuito y Multiplataforma)

Bienvenido al proyecto **Chatbot Académico con RAG (Retrieval-Augmented Generation)**. Este sistema permite a estudiantes de una institución educativa consultar apuntes y materiales oficiales de tres carreras mediante lenguaje natural, obteniendo respuestas precisas respaldadas por citas y fuentes verificables.

---

## 🏛️ Carreras Institucionales Configuradas

1. **Ciberseguridad**
2. **Ciencia de datos**
3. **Desarrollo de software**

---

## 📐 1. Arquitectura del Sistema y Justificación Técnica

### ¿Qué es RAG y por qué lo usamos?
RAG (Generación Aumentada por Recuperación) combina dos procesos:
1. **Recuperación (Retrieval):** Busca en los apuntes locales los fragmentos de texto más parecidos o relevantes a la pregunta del estudiante.
2. **Generación (Generation):** Envía únicamente esos fragmentos junto con la pregunta a un Modelo de Lenguaje (LLM) para que redacte una respuesta basada **estrictamente** en el material institucional.

**Ventajas clave:**
- **Cero alucinaciones:** El modelo no inventa; si la respuesta no está en los apuntes, declara que no hay suficiente información.
- **Gratuito y ligero:** No requiere costosos re-entrenamientos ni servidores GPU dedicados.
- **Transparencia pedagógica:** Cada respuesta incluye las citas exactas (documento, página y fragmento).

---

### Comparativa de Interfaces Gráficas

| Criterio | **CustomTkinter** (Seleccionada) | **Streamlit** | **Gradio** |
| :--- | :--- | :--- | :--- |
| **Tipo de aplicación** | **Escritorio nativo** (Ventana de SO) | Servidor web local + Navegador | Servidor web local + Navegador |
| **Distribución a alumnos** | **Muy sencilla** (Genera `.exe` o `.app` con PyInstaller) | Compleja (Requiere scripts para lanzar servidor y abrir navegador) | Compleja (Requiere servidor web en segundo plano) |
| **Consumo de recursos** | **Muy bajo** (usa el motor de ventanas del SO) | Medio/Alto (Python + Tornado/Tornado web server) | Medio/Alto (Python + FastAPI/Uvicorn) |
| **Conflictos de puertos** | **Ninguno** (No abre puertos de red) | Posibles conflictos de puertos (`localhost:8501`) | Posibles conflictos (`localhost:7860`) |

> **Decisión:** Se seleccionó **CustomTkinter** porque permite entregar al estudiante una aplicación de escritorio tradicional, independiente y empaquetable en un único archivo ejecutable (`.exe` en Windows y `.app` en macOS) sin requerir que abra una pestaña de navegador ni lidie con servidores locales.

---

### Costos y Modelo de Lenguaje: Google Gemini API (Nivel Gratuito)

- **Costo:** \$0.00 USD (100% Gratuito).
- **Límites de la cuenta gratuita (Google AI Studio):**
  - **15 RPM:** Hasta 15 peticiones por minuto.
  - **1,500 RPD:** Hasta 1,500 consultas por día.
  - **1,000,000 TPM:** Hasta 1 millón de tokens por minuto.
- **Seguridad financiera:** La API gratuita en Google AI Studio **no requiere ingresar tarjeta de crédito ni facturación automática**. Si se alcanza el límite de 15 consultas por minuto, la API devuelve un código HTTP `429 (Resource Exhausted)`. La aplicación intercepta este código y le informa amablemente al estudiante que espere unos segundos, sin generar cargos jamás.
- **Alternativa Local (Ollama):** Si en el futuro la institución requiere funcionamiento 100% desconectado de internet, se puede utilizar Ollama con modelos ligeros como `llama3.2:3b`. *Requisitos de hardware para local:* Mínimo 8 GB a 16 GB de RAM y procesador Apple Silicon M1/M2/M3 o tarjeta gráfica NVIDIA en Windows. Para principiantes, la API gratuita de Gemini es más rápida y no satura la memoria del equipo.

---

## 📁 2. Estructura del Proyecto

```text
academic_rag_chatbot/
├── .env.example              # Plantilla de variables de entorno (sin credenciales)
├── .gitignore                # Reglas de exclusión para Git (ignora claves y binarios)
├── requirements.txt          # Dependencias organizadas por etapas
├── README.md                 # Documentación pedagógica y guía de uso
├── main.py                   # Punto de entrada de la aplicación
├── data/                     # Almacenamiento local de datos (modo desarrollo)
│   ├── uploads/              # Carpetas de apuntes organizadas por carrera
│   └── vectordb/             # Almacén de base de datos vectorial
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuración centralizada y gestión de claves
│   ├── gui/                  # Capa de Interfaz Gráfica (CustomTkinter)
│   │   ├── __init__.py
│   │   └── app.py            # Ventana principal, chat, barra lateral y visor de fuentes
│   ├── rag/                  # Capa de Lógica de Recuperación y Generación
│   │   ├── __init__.py
│   │   └── service.py        # Servicio RAG (Mock funcional en Etapa 1)
│   └── utils/                # Utilidades de soporte
│       ├── __init__.py
│       └── paths.py          # Resolución de rutas multiplataforma (macOS/Windows)
└── tests/
    └── test_stage1.py        # Pruebas automatizadas de la Etapa 1
```

---

## 🚀 3. Instrucciones de Instalación y Ejecución

### Prerrequisitos
- **Python 3.10 o superior** (probado en Python 3.13 en macOS Apple Silicon).

### Paso 1: Clonar o ingresar al directorio del proyecto
```bash
cd /Users/diazagustin/.gemini/antigravity/scratch/academic_rag_chatbot
```

### Paso 2: Crear y activar el entorno virtual
En macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Paso 3: Instalar las dependencias de la Etapa 1
```bash
pip install -r requirements.txt
```

### Paso 4: Ejecutar la aplicación
Para iniciar la interfaz gráfica interactiva:
```bash
python main.py
```

Para ejecutar la verificación automatizada rápida (sin dejar la ventana abierta):
```bash
python main.py --test-gui
```

Para correr las pruebas unitarias:
```bash
python -m unittest tests/test_stage1.py
```

---

## 🔑 4. Cómo Configurar la Clave de Google Gemini (Gratis)

1. Ingresa a [Google AI Studio](https://aistudio.google.com/).
2. Inicia sesión con una cuenta de Google normal.
3. Haz clic en el botón **"Get API key"** y luego en **"Create API key"**.
4. Copia tu clave generada.
5. Puedes configurarla de dos formas:
   - **Desde la aplicación:** Inicia la app, haz clic en el botón **"🔑 Configurar Clave API"** en la barra lateral e ingresa tu clave. Se guardará de forma privada en tu archivo `.env`.
   - **Manualmente:** Copia `.env.example` como `.env` y asigna `GEMINI_API_KEY=tu_clave_aqui`.

> ⚠️ **Seguridad:** El archivo `.env` está incluido en `.gitignore`. Nunca compartas tu clave ni la subas a repositorios públicos.

---

## 🧪 5. Verificaciones Realizadas

### Etapa 1: Base del Sistema e Interfaz
- [x] **Compatibilidad de Python y Tkinter:** Verificado en macOS Apple Silicon con Tkinter 8.6.
- [x] **Instalación de librerías base:** `customtkinter` y `python-dotenv` operativas.
- [x] **Rutas multiplataforma:** La función `get_app_data_dir()` resuelve correctamente las rutas en macOS, Windows y desarrollo.
- [x] **Interfaz gráfica funcional:** Selector de carreras, chat con historial, visor de fuentes y configuración de API.

### Etapa 2: Extracción y Gestión Documental
- [x] **Nuevas carreras institucionales:** Ciberseguridad, Ciencia de datos y Desarrollo de software.
- [x] **Extracción de PDF y TXT:** Módulo `DocumentExtractor` usando `pypdf` que extrae texto y preserva el número de página (1-indexed).
- [x] **Deduplicación por Hash SHA-256:** `DocumentManager` calcula el hash del archivo y rechaza documentos duplicados (incluso con distinto nombre).
- [x] **Persistencia y Organización:** Copia física estructurada en `data/uploads/<carrera>/` y metadatos persistentes en `data/documents_meta.json`.
- [x] **Eliminación individual:** Botón `❌` en la barra lateral para eliminar documentos individuales o botón para vaciar todo.
- [x] **Pruebas Automatizadas:** 12/12 pruebas unitarias superadas (`tests/test_stage1.py` y `tests/test_stage2.py`).

### Etapa 3: Embeddings, ChromaDB y Recuperación Semántica
- [x] **Fragmentación Inteligente (Chunking):** Módulo `TextChunker` con solapamiento (`chunk_size=600`, `overlap=120`) y cortes en párrafos y oraciones completas.
- [x] **Base Vectorial Local Persistente:** Módulo `VectorStore` con **ChromaDB** en `data/vectordb/`.
- [x] **Embeddings Locales Gratuitos:** Modelo ONNX `all-MiniLM-L6-v2` (384 dimensiones) que corre 100% desconectado en CPU/Apple Silicon.
- [x] **Búsqueda Semántica con Filtro de Carrera:** Búsqueda por distancia coseno combinada con cláusula `where={"career": ...}`.
- [x] **Reconocimiento de Información Insuficiente (Requisito 7):** Calibración de umbral de similitud mínima para responder honestamente cuando un tema no figura en los apuntes.
- [x] **Eliminación Sincronizada:** Al borrar un documento, sus fragmentos vectoriales se purgan de ChromaDB.
- [x] **Pruebas Automatizadas:** 18/18 pruebas unitarias superadas (`test_stage1`, `test_stage2` y `test_stage3`).

### Etapa 4: Generación con Google Gemini y Manejo Seguro de Errores
- [x] **SDK Oficial de Google GenAI:** Integración de la biblioteca `google-genai` con soporte para `gemini-1.5-flash` y `gemini-2.0-flash`.
- [x] **Prompt Académico Estricto:** Instrucciones del sistema que obligan a responder con tono pedagógico basándose **únicamente** en los fragmentos recuperados y citando documentos y páginas.
- [x] **Gestión Preventiva de Cuotas Gratuitas (HTTP 429):** Captura amigable del error de límite por minuto (15 RPM) sin generar costos ni intentar cobros automáticos.
- [x] **Manejo de Errores de Autenticación y Conectividad:** Respuestas claras ante claves API erróneas, caídas de red o falta de internet.
- [x] **Modo Offline / Fallback Local:** Si no se dispone de clave o conexión, la aplicación presenta los fragmentos locales recuperados en la Etapa 3 sin colapsar.
- [x] **Pruebas Automatizadas:** 24/24 pruebas unitarias superadas (`test_stage1`, `test_stage2`, `test_stage3` y `test_stage4`).

### Etapa 5: Citas, Fuentes y Verificación de Cero Alucinaciones
- [x] **Módulo de Verificación de Citas:** Clase `CitationVerifier` que extrae las citas del texto (`[Documento, Pág.]`) y valida su correspondencia contra los fragmentos reales de ChromaDB.
- [x] **Batería de Cero Alucinaciones:** Comprobado el rechazo estricto ante preguntas con premisas falsas y preguntas fuera del catálogo académico.
- [x] **Interfaz Enriquecida de Fuentes:** Tarjetas interactivas con indicador de porcentaje de relevancia, botón para copiar fragmento al portapapeles y badge de verificación `🛡️ Cero Alucinaciones`.
- [x] **Pruebas Automatizadas:** 30/30 pruebas unitarias superadas (`test_stage1` a `test_stage5`).

---

## ⏭️ Próximos Pasos: Etapa 6 (OCR Gratuito para Imágenes y Administración Avanzada)

En la siguiente etapa implementaremos:
1. **OCR Gratuito para Imágenes con Texto:** Soporte para importar imágenes (`.png`, `.jpg`, `.jpeg`) de pizarras o apuntes manuscritos/fotocopiados mediante **Tesseract OCR / Pillow**.
2. **Administración Documental Refinada:** Filtros avanzados por carrera y estadísticas de almacenamiento.
3. **Manejo de Errores de Imagen:** Diagnóstico de imágenes sin texto o de baja resolución.



