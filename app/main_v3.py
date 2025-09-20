import os
import sys
import json
import logging
import yaml
import itertools
import numpy as np
from datetime import datetime
from collections import defaultdict
import yaml
import numpy as np
import itertools
from utils.logger import setup_logging
from dotenv import load_dotenv
# Cargar variables de entorno desde .env
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', '.env')))
# Obtener timestamp actual
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Crear carpetas de logs y resultados con timestamp
log_dir = os.path.join("logs", timestamp)
results_dir = os.path.join("results", timestamp)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
# Establece el valor global
from utils.loader import send_question_to_target
from app.evaluator import evaluate_response
from utils.change_settins import update_webapp_settings

# Configurar logging con archivo en carpeta correspondiente
setup_logging()
logger = logging.getLogger("main")
logger.info("Inicio del programa principal")

def check_env_vars(var_names):
    missing_vars = []
    values = {}
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

def calcular_promedio_metricas(scores):
    if not scores:
        return {}

    promedio = defaultdict(float)
    total = len(scores)

    for score in scores:
        for k, v in score.items():
            if isinstance(v, (int, float)):
                promedio[k] += v

    for k in promedio:
        promedio[k] = round(promedio[k] / total, 3)

    return dict(promedio)

def calcular_score_global_unificado(scores_por_pregunta):
    puntuaciones = [s["weighted_score"] for s in scores_por_pregunta if s.get("weighted_score") is not None]
    if not puntuaciones:
        return None
    return round(sum(puntuaciones) / len(puntuaciones), 4)

def calcular_score_ponderado(score_dict, label, pesos_por_etiqueta):
    pesos = pesos_por_etiqueta.get(label, {})
    if not pesos:
        return None

    score_total = 0.0
    peso_total = 0.0

    for metrica, peso in pesos.items():
        valor_original = score_dict.get(metrica)
        if valor_original is not None:
            score_total += valor_original * peso
            peso_total += peso

    if peso_total == 0:
        return None
    return round(score_total / peso_total, 4)

def normalizar_metrica(nombre, valor):
    if nombre == "f1_score":
        return valor
    elif nombre.startswith("gpt_") or nombre in [
        "similarity", "relevance", "groundedness", "coherence", "fluency"
    ]:
        return max(0.0, min((valor - 1) / 4, 1.0))
    return None

def generar_combinaciones_parametros(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    samples = {}
    for param, valores in config.items():
        if isinstance(valores, list):
            # Parámetros con valores discretos (como mensajes)
            samples[param] = valores
        elif isinstance(valores, dict) and "min" in valores and "max" in valores:
            tipo = valores.get("type", "float")
            # Generar 4 muestras equiespaciadas
            muestras = np.linspace(valores["min"], valores["max"], 4)
            if tipo == "int":
                muestras = [int(round(v)) for v in muestras]
            else:
                muestras = [round(float(v), 3) for v in muestras]
            samples[param] = muestras

    claves = list(samples.keys())
    valores = list(samples.values())
    combinaciones = [dict(zip(claves, combo)) for combo in itertools.product(*valores)]
    return combinaciones


def main():
    try:
        logger.info("Inicio del proceso de evaluación del chatbot.")

        env_vars = check_env_vars(["SUBSCRIPTION_ID", "RESOURCE_GROUP", "WEB_APP_NAME"])
        subscription_id = env_vars["SUBSCRIPTION_ID"]
        resource_group = env_vars["RESOURCE_GROUP"]
        webapp_name = env_vars["WEB_APP_NAME"]

        with open("data/questions.json", "r", encoding="utf-8") as f:
            questions_by_label = json.load(f)

        with open("configs/metric_weights.yaml", "r", encoding="utf-8") as f:
            metric_weights_by_label = yaml.safe_load(f)["weights"]

        config_path = "configs/settings_ranges.yaml"
        combinaciones = generar_combinaciones_parametros(config_path)

        archivo_pendientes = os.path.join(results_dir, "combinaciones_pendientes.json")
        archivo_realizadas = os.path.join(results_dir, "combinaciones_realizadas.json")

        if os.path.exists(archivo_pendientes):
            with open(archivo_pendientes, "r", encoding="utf-8") as f:
                combinaciones = json.load(f)

        realizadas = []

        for idx, config in enumerate(combinaciones):
            logger.info(f"Aplicando configuración {idx+1}/{len(combinaciones)}: {config}")
            update_webapp_settings(
                subscription_id=subscription_id, 
                resource_group=resource_group, 
                webapp_name=webapp_name,
                new_settings=config
)
            logger.info(f"Evaluando configuración {idx+1}/{len(combinaciones)}: {config}")

            auth_mode = "cookies"
            all_scores = []
            label_scores = defaultdict(list)
            for qid, label, question, expected, actual, context in send_question_to_target(questions_by_label, auth_mode):
                if actual is None:
                    logger.error(f"No se recibió respuesta para la pregunta {qid}")
                    continue

                score = evaluate_response(
                    qid=qid,
                    label=label,
                    query=question,
                    response=actual,
                    context=context,
                    ground_truth=expected
                )

                weighted_score = calcular_score_ponderado(score, label, metric_weights_by_label)

                score_entry = {
                    "id": qid,
                    "label": label,
                    "question": question,
                    "expected": expected,
                    "actual": actual,
                    "context": context,
                    "score": score,
                    "weighted_score": weighted_score
                }

                all_scores.append(score_entry)
                label_scores[label].append(score)

            resumen = {
                "config": config,
                "resultados": all_scores
            }

            with open(os.path.join(results_dir, f"scores_config_{idx+1}.json"), "w", encoding="utf-8") as f:
                json.dump(resumen, f, indent=2, ensure_ascii=False)

            realizadas.append(config)
            pendientes = combinaciones[idx+1:]

            with open(archivo_pendientes, "w", encoding="utf-8") as f:
                json.dump(pendientes, f, indent=2)

            with open(archivo_realizadas, "w", encoding="utf-8") as f:
                json.dump(realizadas, f, indent=2)

        print(f"✅ Evaluación completada. Revisa resultados en: {results_dir}")
        logger.info("Evaluación finalizada exitosamente.")

    except Exception as e:
        logger.exception(f"Error en la ejecución principal: {str(e)}")

if __name__ == "__main__":
    main()