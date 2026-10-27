"""
Módulo de Integración con Google Gemini API (Modalidad Gratuita).

Implementa:
1. Conexión segura con Google Gemini API mediante el SDK oficial google-genai.
2. Prompt estricto que restringe la respuesta únicamente al material institucional recuperado.
3. Manejo preventivo de cuotas gratuitas (HTTP 429 - 15 RPM) sin generar costos.
4. Modo degradado / Fallback local cuando no hay clave o no hay conexión a internet.
"""

from typing import List, Dict, Any, Optional
import httpx

from src.config import Config

try:
    from google import genai
    from google.genai import types, errors
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class LLMClient:
    """Cliente para la generación de respuestas pedagógicas basadas en RAG."""

    SYSTEM_INSTRUCTION = (
        "Eres un asistente pedagógico y profesor universitario de la institución educativa. "
        "Tu objetivo es responder la consulta del estudiante utilizando EXCLUSIVAMENTE los fragmentos de apuntes oficiales proporcionados en el contexto.\n\n"
        "REGLAS OBLIGATORIAS:\n"
        "1. Responde en español con tono pedagógico, claro y bien estructurado (utiliza listas o viñetas cuando facilite el aprendizaje).\n"
        "2. No inventes información ni utilices conocimiento externo que contradiga o no esté sustentado en los fragmentos.\n"
        "3. Cita explícitamente en el cuerpo de la respuesta el documento y la página donde se sustenta cada afirmación (ejemplo: '[Documento: X, Pág. Y]').\n"
        "4. Si los fragmentos proporcionados NO contienen información suficiente para responder a la pregunta, debes responder con honestidad: 'Los materiales cargados en el sistema no contienen información suficiente para responder a esta consulta.' No intentes adivinar ni completar con datos no verificables."
    )

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model_name = model_name or Config.GEMINI_MODEL
        self._client: Optional[Any] = None

    def _get_client(self) -> Optional[Any]:
        """Inicializa o actualiza el cliente de Google GenAI con la clave configurada."""
        # Re-verificar la clave en Config por si el usuario la configuró desde la interfaz
        current_key = Config.GEMINI_API_KEY or self.api_key
        if not current_key:
            return None

        if self._client is None or self.api_key != current_key:
            self.api_key = current_key
            if GENAI_AVAILABLE:
                self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _format_context(context_chunks: List[Dict[str, Any]]) -> str:
        """Formatea los fragmentos recuperados para su inclusión en el prompt."""
        if not context_chunks:
            return "No se recuperaron fragmentos relevantes."

        formatted_pieces = []
        for idx, chunk in enumerate(context_chunks, 1):
            doc = chunk.get("document", "Documento")
            page = chunk.get("page", 1)
            career = chunk.get("career", "N/A")
            text = chunk.get("fragment", "")
            formatted_pieces.append(
                f"--- FRAGMENTO #{idx} ---\n"
                f"Documento: {doc} | Página: {page} | Carrera: {career}\n"
                f"Contenido:\n{text}\n"
            )
        return "\n".join(formatted_pieces)

    def generate_answer(
        self,
        query: str,
        career: str,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Genera la respuesta pedagógica utilizando Gemini API.
        Si la API no está disponible, no hay clave o hay error de red,
        devuelve una respuesta estructurada con los fragmentos locales (Modo Offline).
        """
        context_text = self._format_context(context_chunks)
        client = self._get_client()

        # 1. Caso: Clave API no configurada
        if not client:
            fallback_answer = (
                f"ℹ️ [Modo Local - Sin Clave API]:\n\n"
                f"Se recuperaron {len(context_chunks)} fragmentos relevantes de los apuntes para tu consulta, "
                f"pero aún no has configurado tu clave gratuita de Google Gemini.\n\n"
                f"👉 Para que la Inteligencia Artificial redacte una explicación completa y didáctica:\n"
                f"1. Haz clic en el botón '🔑 Configurar Clave API' en la barra lateral.\n"
                f"2. Pega tu clave gratuita de Google AI Studio (https://aistudio.google.com/).\n\n"
                f"Mientras tanto, puedes leer los fragmentos recuperados en el panel derecho de fuentes."
            )
            return {
                "answer": fallback_answer,
                "used_llm": False,
                "error": None
            }

        # 2. Caso: Consulta real a Google Gemini API
        prompt_content = (
            f"Carrera Institucional: {career}\n"
            f"Consulta del estudiante: \"{query}\"\n\n"
            f"Fragmentos de apuntes oficiales recuperados:\n"
            f"{context_text}\n\n"
            f"Por favor, redacta la respuesta para el estudiante siguiendo estrictamente las instrucciones pedagógicas."
        )

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt_content,
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_INSTRUCTION,
                    temperature=0.2,  # Baja temperatura para evitar alucinaciones y ser fiel a los apuntes
                    max_output_tokens=1000
                )
            )

            answer_text = response.text or "El modelo no generó texto de respuesta."
            return {
                "answer": answer_text,
                "used_llm": True,
                "error": None
            }

        except errors.APIError as e:
            # Manejo específico del límite de cuota gratuita (HTTP 429)
            if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e).upper():
                msg = (
                    "⚠️ Límite gratuito temporal alcanzado (HTTP 429 - Cuota de peticiones por minuto).\n\n"
                    "La modalidad gratuita de Google Gemini en Google AI Studio permite hasta 15 consultas por minuto.\n"
                    "Por favor espera unos 30 segundos antes de enviar una nueva consulta.\n\n"
                    "🔒 Seguridad financiera: Tu cuenta es 100% gratuita y no se generará ningún cargo."
                )
                return {"answer": msg, "used_llm": False, "error": "RATE_LIMIT_429"}

            # Clave inválida o error de permisos
            if getattr(e, "code", None) in (400, 401, 403) or "API_KEY_INVALID" in str(e).upper():
                msg = (
                    "❌ Error de autenticación con Google Gemini (Clave no válida o sin permisos).\n\n"
                    "Por favor verifica tu clave gratuita en '🔑 Configurar Clave API' o genera una nueva en https://aistudio.google.com/."
                )
                return {"answer": msg, "used_llm": False, "error": "INVALID_API_KEY"}

            # Otro error de API
            msg = f"❌ Error de la API de Gemini ({getattr(e, 'code', 'desconocido')}): {e.message}"
            return {"answer": msg, "used_llm": False, "error": str(e)}

        except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
            # Error de conectividad de red / desconectado
            msg = (
                "⚠️ Error de conexión a internet.\n\n"
                "No se pudo establecer contacto con los servidores de Google Gemini. "
                "Verifica tu conexión a la red.\n\n"
                "(Los fragmentos recuperados localmente siguen disponibles en el panel derecho)."
            )
            return {"answer": msg, "used_llm": False, "error": "NETWORK_ERROR"}

        except Exception as e:
            msg = f"❌ Ocurrió un error inesperado al consultar el modelo de lenguaje: {e}"
            return {"answer": msg, "used_llm": False, "error": str(e)}
