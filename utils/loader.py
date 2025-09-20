# utils/loader.py

import os
import json
import logging
from typing import Optional, Dict
import requests
import sys

logger = logging.getLogger("loader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)

def check_env_vars(var_names):
    missing_vars, values = [], {}
    for var in var_names:
        val = os.getenv(var)
        if val is None:
            missing_vars.append(var)
        else:
            values[var] = val
    if missing_vars:
        msg = f"Faltan las variables de entorno: {', '.join(missing_vars)}"
        logging.error(msg)
        sys.exit(f"Error: {msg}. Revisa error.log")
    return values

def get_session_cookies():
    # Validamos cookies SOLO si se usa el modo cookies
    cookie_vars = check_env_vars(["ARR_AFFINITY", "ARR_AFFINITY_SAMESITE", "APP_SERVICE_AUTH_SESSION"])
    return {
        "ARRAffinity": cookie_vars["ARR_AFFINITY"],
        "ARRAffinitySameSite": cookie_vars["ARR_AFFINITY_SAMESITE"],
        "AppServiceAuthSession": cookie_vars["APP_SERVICE_AUTH_SESSION"]
    }

# Solo exigimos WEBCHAT_URL al cargar; PUBLIC_API_URL puede no existir si no usas "none"
env_vars = check_env_vars(["WEBCHAT_URL"])

def _pick_target_url(auth_mode: str) -> str:
    """
    cookies / client_secret -> WEBCHAT_URL (tu proxy protegido)
    none -> PUBLIC_API_URL (endpoint público de API que acepte POST)
    """
    if auth_mode == "none":
        public = os.getenv("PUBLIC_API_URL")
        if not public:
            raise RuntimeError(
                "Para auth_mode='none' debes definir PUBLIC_API_URL apuntando al endpoint de API que acepta POST."
            )
        return public
    # cookies o client_secret
    return env_vars["WEBCHAT_URL"]

def send_question_to_target(
    question: str,
    parameters: dict = {},
    auth_mode: str = "cookies",   # "cookies" | "client_secret" | "none"
    token: Optional[str] = None
) -> dict:
    headers = {
        "Content-Type": "application/json",
        # Si tu backend streamea, esto ayuda; si no, no molesta.
        "Accept": "text/event-stream, application/json"
    }
    body = {
        "messages": [{"role": "user", "content": question}],
        "context": parameters
    }

    # ⬅️ ESTA ES LA CLAVE: elegir URL según el modo
    target_url = _pick_target_url(auth_mode)

    try:
        if auth_mode == "cookies":
            session_cookies = get_session_cookies()
            logger.debug("Enviando solicitud con cookies.")
            response = requests.post(target_url, headers=headers, json=body, cookies=session_cookies, verify=False, timeout=120)

        elif auth_mode == "client_secret":
            if not token:
                logger.error("Token no proporcionado para 'client_secret'.")
                raise ValueError("Token no proporcionado para 'client_secret'.")
            headers["Authorization"] = f"Bearer {token}"
            logger.debug("Enviando solicitud con client_secret.")
            response = requests.post(target_url, headers=headers, json=body, verify=False, timeout=120)

        elif auth_mode == "none":
            # Sin autenticación
            logger.debug("Enviando solicitud sin autenticación (none).")
            response = requests.post(target_url, headers=headers, json=body, verify=False, timeout=240)

        else:
            logger.error(f"Modo de autenticación no soportado: {auth_mode}")
            raise ValueError(f"Modo de autenticación no soportado: {auth_mode}")

        latency = response.elapsed.total_seconds()
        logger.info(f"Latencia: {latency:.3f} s")

        if response.status_code == 405:
            allow = response.headers.get("Allow", "<desconocido>")
            logger.error(f"405 Method Not Allowed. Métodos permitidos por el servidor: {allow}")
            logger.error(f"URL usada: {target_url}")
            raise RuntimeError(f"Error HTTP 405 (Allow: {allow}). Revisa que la URL sea el endpoint POST correcto.")

        if response.status_code == 401:
            logger.error("Error 401: No autorizado.")
            raise PermissionError("Error 401: No autorizado.")

        if not response.ok:
            logger.error(f"Error HTTP {response.status_code}: {response.reason}")
            raise RuntimeError(f"Error HTTP {response.status_code}: {response.reason}")

        response.encoding = "utf-8"
        content_txt = ""
        context_txt = ""

        for line in response.text.splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                messages = data.get("choices", [{}])[0].get("messages", [])
                for msg in messages:
                    if msg.get("role") == "assistant":
                        content_txt += msg.get("content", "")
                    elif msg.get("role") == "tool":
                        context_txt += msg.get("content", "")
            except json.JSONDecodeError as e:
                logger.warning(f"Error decodificando JSON: {e}")
                # Si tu API devuelve un único JSON, intenta parsear el cuerpo completo:
                try:
                    data = json.loads(response.text)
                    messages = data.get("choices", [{}])[0].get("messages", [])
                    for msg in messages:
                        if msg.get("role") == "assistant":
                            content_txt += msg.get("content", "")
                        elif msg.get("role") == "tool":
                            context_txt += msg.get("content", "")
                except json.JSONDecodeError:
                    # Devuelve el texto crudo para depurar
                    return {"answer": response.text.strip(), "context": None, "latency": latency}

        return {
            "answer": content_txt.strip(),
            "context": context_txt.strip() if context_txt else None,
            "latency": latency
        }

    except requests.exceptions.RequestException as e:
        logger.exception("Error de conexión con el servidor")
        raise e
    except Exception as e:
        logger.exception("Error inesperado al enviar la pregunta")
        raise e