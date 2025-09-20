# utils/loader.py
import os
from dotenv import load_dotenv
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', '.env')))
import json
import logging
import requests
from typing import Dict, List, Tuple, Generator


# Configurar logging
logging.basicConfig(
    filename="logs/loader.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def get_session_cookies():
    return {
        "ARRAffinity": os.environ["ARR_AFFINITY"],
        "ARRAffinitySameSite": os.environ["ARR_AFFINITY_SAMESITE"],
        "AppServiceAuthSession": os.environ["APP_SERVICE_AUTH_SESSION"]
    }

from typing import Optional

def send_question_to_target(
    question: str,
    url: str,
    parameters: dict = {},
    auth_mode: str = "cookies",
    token: Optional[str] = None
):
    headers = {"Content-Type": "application/json"}
    body = {
        "messages": [{"content": question, "role": "user"}],
        "context": parameters,
    }

    try:
        print("Enviando pregunta:", question)  # <-- Agrega esto
        if auth_mode == "cookies":
            session_cookies = get_session_cookies()
            r = requests.post(url, headers=headers, json=body, cookies=session_cookies, verify=False)
            print("respuesta sin procesar:", r)  # <-- Agrega esto
        elif auth_mode == "client_secret":
            if token is None:
                raise ValueError("Token is required for client_secret authentication")
            headers["Authorization"] = f"Bearer {token}"
            r = requests.post(url, headers=headers, json=body, verify=False)
        else:
            raise ValueError("Unsupported authentication mode")

        r.encoding = "utf-8"
        latency = r.elapsed.total_seconds()
        response_text = r.text
        print("Respuesta cruda del servidor:", response_text)  # <-- Agrega esto

        content_txt = ""
        context_txt = ""

        for line in response_text.splitlines():
            try:
                if not line.strip():
                    continue
                data = json.loads(line)
                messages = data.get("choices", [{}])[0].get("messages", [])
                for message in messages:
                    role = message.get("role")
                    content = message.get("content", "")
                    if role == "assistant":
                        content_txt += content
                    elif role == "tool":
                        context_txt += content
            except AttributeError as e:
                print("Error accediendo a atributos:", e)

        answer = content_txt.strip()
        context = context_txt.strip() if context_txt else None
        return {
            "answer": answer,
            "context": context,
            "latency": latency
        }

    except Exception as e:
        return {
            "answer": str(e),
            "context": str(e),
            "latency": -1,
        }

def authenticate_with_client_secret(auth_url: str, client_id: str, client_secret: str, scope: str) -> str:
    try:
        payload = {
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope,
            "grant_type": "client_credentials"
        }
        response = requests.post(auth_url, data=payload)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        logging.error(f"Client secret authentication failed: {e}")
        return ""

def response_generator(
    questions_by_label: Dict[str, List[Dict]],
    auth_mode: str,
) -> Generator[Tuple[str, str, str, str, str, str], None, None]:
    token = None

    if auth_mode == "client_secret":
    #    token = authenticate_with_client_secret(
    #        config["api_url"],
    #        config["client_id"],  
  #)
        if not token:
            logging.error("No se pudo obtener el token para client_secret.")
            return

    # Selecciona la URL correcta según el modo de autenticación
    url = os.environ["WEBCHAT_URL"] if auth_mode == "cookies" else os.environ["api_url"]
    print("URL de destino:", url)  # <-- Agrega esto
    for label, questions in questions_by_label.items():
        for q in questions:
            qid = str(q.get("id") or "")
            question_text = str(q.get("question") or "None")
            expected = str(q.get("expected_answer") or "")
            result = send_question_to_target(
                question=question_text,
                url=url,
                parameters={},
                auth_mode=auth_mode,
                token=token
            )
            actual = str(result.get("answer", "") or "")
            context = str(result.get("context", "") or "")
            yield qid, label, question_text, expected, actual, context


# Ejemplo de uso para probar
if __name__ == "__main__":

    print( os.environ["WEBCHAT_URL"])
    questions_by_label = {
        "general": [
            {
                "id": "q1",
                "question": "¿Cuál es la capital de Francia?",
                "expected_answer": "París"
            },
            {
                "id": "q2",
                "question": "¿Cuál es la capital de Alemania?",
                "expected_answer": "Berlín"
            }
        ]
    }

    config = {
        "WEBCHAT_URL": os.environ["WEBCHAT_URL"],
        "api_url": "https://tu-api-url.com/endpoint",  # Cambia esto por tu endpoint real
        "auth_url": "https://tu-api-url.com/auth",
        "client_id": os.environ.get("CLIENT_ID", ""),
        "client_secret": os.environ.get("CLIENT_SECRET", ""),
        "scope": "api://default/.default"
    }

    auth_mode = "cookies"  # O "client_secret"

    for qid, label, question, expected, actual, context in response_generator(questions_by_label, auth_mode):
        print(f"ID: {qid}")
        print(f"Etiqueta: {label}")
        print(f"Pregunta: {question}")
        print(f"Esperado: {expected}")
        print(f"Respuesta: {actual}")
        print(f"Contexto: {context}")
        print("-" * 40)
