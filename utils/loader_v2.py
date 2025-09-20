# utils/loader.py

import os
import sys
import json
import logging
from typing import Optional, Dict
import requests

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("loader")
logger.setLevel(logging.INFO)

# Asegura que exista un handler (evita duplicados si se importa varias veces)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

logger.info("loader.py inicializado")

# -----------------------------
# Utilidades
# -----------------------------
def check_env_vars(var_names) -> Dict[str, str]:
    """Devuelve dict con las vars existentes; si falta alguna, aborta."""
    missing, values = [], {}
    for var in var_names:
        val = os.getenv(var)
        if val is None:
            missing.append(var)
        else:
            values[var] = val
    if missing:
        msg = f"Faltan las variables de entorno: {', '.join(missing)}"
        logger.error(msg)
        sys.exit(f"Error: {msg}")
    return values

def get_session_cookies() -> Dict[str, str]:
    """Obtiene cookies de sesión para modo 'cookies'. Valida que existan."""
    cookie_vars = check_env_vars(["ARR_AFFINITY", "ARR_AFFINITY_SAMESITE", "APP_SERVICE_AUTH_SESSION"])
    return {
        "ARRAffinity": cookie_vars["ARR_AFFINITY"],
        "ARRAffinitySameSite": cookie_vars["ARR_AFFINITY_SAMESITE"],
        "AppServiceAuthSession": cookie_vars["APP_SERVICE_AUTH_SESSION"],
    }

# URL base requerida para cualquier modo (si quieres una pública separada, usa PUBLIC_API_URL y pásala como url)
_BASE_VARS = check_env_vars(["WEBCHAT_URL"])

# -----------------------------
# Principal
# -----------------------------
def send_question_to_target(
    question: str,
    parameters: dict = {},
    auth_mode: str = "cookies",   # "cookies" | "client_secret" | "none"
    token: Optional[str] = None,
    url: Optional[str] = None,
    timeout: int = 120,
    verify_ssl: bool = False,     # mantenido en False para reproducir tu comportamiento actual
) -> dict:
    """
    Envía la pregunta al endpoint configurado. Soporta 3 modos:
      - "cookies": usa cookies de App Service
      - "client_secret": usa Authorization: Bearer <token>
      - "none": sin autenticación
    """
    target_url = url or _BASE_VARS["WEBCHAT_URL"]
    headers = {"Content-Type": "application/json"}
    body = {
        "messages": [{"role": "user", "content": question}]
        # Nota: 'parameters' está disponible si tu backend lo requiere; actualmente no lo añades al body.
        # Si lo necesitas, descomenta la línea siguiente:
        # , "context": parameters
    }

    try:
        logger.info(f"Enviando pregunta (modo={auth_mode}) a {target_url}")

        if auth_mode == "cookies":
            session_cookies = get_session_cookies()
            response = requests.post(
                target_url, headers=headers, json=body,
                cookies=session_cookies, verify=verify_ssl, timeout=timeout
            )

        elif auth_mode == "client_secret":
            if not token:
                logger.error("Token no proporcionado para 'client_secret'.")
                raise ValueError("Token no proporcionado para 'client_secret'.")
            headers["Authorization"] = f"Bearer {token}"
            response = requests.post(
                target_url, headers=headers, json=body,
                verify=verify_ssl, timeout=timeout
            )

        elif auth_mode == "none":
            # Sin cookies, sin Authorization
            response = requests.post(
                target_url, headers=headers, json=body,
                verify=verify_ssl, timeout=timeout
            )

        else:
            logger.error(f"Modo de autenticación no soportado: {auth_mode}")
            raise ValueError(f"Modo de autenticación no soportado: {auth_mode}")

        latency = response.elapsed.total_seconds() if response is not None else -1.0
        logger.info(f"Latencia: {latency:.3f} s")

        if response.status_code == 401:
            logger.error("Error 401: No autorizado.")
            raise PermissionError("Error 401: No autorizado.")

        if not response.ok:
            logger.error(f"Error HTTP {response.status_code}: {response.reason}")
            raise RuntimeError(f"Error HTTP {response.status_code}: {response.reason}")

        response.encoding = "utf-8"

        # Parseo estilo JSONL (una línea por fragmento)
        content_txt = ""
        context_txt = ""

        for line in response.text.splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                messages = data.get("choices", [{}])[0].get("messages", [])
                for msg in messages:
                    role = msg.get("role")
                    if role == "assistant":
                        content_txt += msg.get("content", "")
                    elif role == "tool":
                        context_txt += msg.get("content", "")
            except json.JSONDecodeError as e:
                logger.warning(f"Error decodificando JSON por líneas: {e}")
                # Si tu backend no envía JSON por líneas, podrías intentar parsear todo el cuerpo:
                # try:
                #     data = json.loads(response.text)
                #     # ... procesar igual que arriba
                # except json.JSONDecodeError:
                #     raise
                raise

        return {
            "answer": content_txt.strip(),
            "context": context_txt.strip() if context_txt else None,
            "latency": latency
        }

    except requests.exceptions.RequestException as e:
        logger.exception("Error de conexión con el servidor")
        raise
    except Exception as e:
        logger.exception("Error inesperado al enviar la pregunta")
        raise

# -----------------------------
# Ejemplo rápido (descomentable)
# -----------------------------
"""
if __name__ == "__main__":
    try:
        res = send_question_to_target(
            question="¿Qué es un elevador?",
            auth_mode="none"  # "cookies" | "client_secret" | "none"
            # token="...",     # si usas client_secre
            # url=os.getenv("PUBLIC_API_URL")  # si quieres usar otra URL pública
        )
        print(res)
    except Exception as e:
        print("Fallo:", e)
"""
