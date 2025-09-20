# utils/evaluator.py
import os
import logging
from azure.ai.evaluation import AzureOpenAIModelConfiguration, QAEvaluator
from typing import Dict
import sys
import requests



# Configurar logging con archivo en carpeta correspondiente
logger = logging.getLogger("evaluator")
logger.info("Este es un log desde evaluator.py")

model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["MODEL_DEPLOYMENT_NAME"],
)


def evaluate_response(
    qid: str,
    label: str,
    query: str,
    response: str,
    context: str,
    ground_truth: str,
) -> Dict:
    try:
        logger.info(f"[{qid}] Creando QAEvaluator con model_config...")
        evaluator = QAEvaluator(model_config)
        
        logger.info(f"[{qid}] QAEvaluator creado correctamente. Iniciando evaluación...")
        result = evaluator(
            query=query,
            response=response,
            context=context,
            ground_truth=ground_truth,
        )

        if not result or not isinstance(result, dict):
            raise ValueError("Evaluador devolvió una respuesta vacía o inválida.")

        logger.info(f"[{qid}] Evaluación completada correctamente.")
        result["id"] = qid
        result["label"] = label
        return result

    except requests.exceptions.RequestException as e:
        logger.error(f"[{qid}] Error de conexión al servidor durante la evaluación: {e}")
        raise e

    except PermissionError as e:
        logger.error(f"[{qid}] Error de autenticación durante la evaluación: {e}")
        raise e  # propaga el error a .main

    except Exception as e:
        logger.exception(f"[{qid}] Fallo inesperado durante la evaluación: {e}")
        raise e  # Deja que .main lo capture y lo reporte por email

"""""

# Datos de prueba
qid = "test1"
label = "correct"
query = "¿Cuál es la capital de Francia?"
response = "La capital de Francia es París."
context = "Francia es un país en Europa. Su capital es París."
ground_truth = "París"

# Ejecutar la función
result = evaluate_response(
    qid=qid,
    label=label,
    query=query,
    response=response,
    context=context,
    ground_truth=ground_truth
)

print("Resultado de la evaluación:", result)
"""

