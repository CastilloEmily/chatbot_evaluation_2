import os
import sys
import json, re
import logging
import yaml
import math
from functools import lru_cache
from datetime import datetime
from itertools import product
import numpy as np
from utils.logger import setup_logging
import time
from dotenv import load_dotenv
from pathlib import Path
import argparse
# Cargar variables de entorno desde .env
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', '.env')))
# Importar funciones de otros módulos
from utils.loader import send_question_to_target
from app.evaluator import evaluate_response
from utils.change_settins import update_webapp_settings
from utils.create_report import EmailReporter

# Configurar logging con archivo en carpeta correspondiente
setup_logging()
logger = logging.getLogger("main")
logger.info("Inicio del programa principal")

# Crear carpetas de resultados con timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Crear carpetas de logs y resultados con timestamp
results_dir = os.path.join("results", timestamp)
os.makedirs(results_dir, exist_ok=True)
output_path_questions = os.path.join(results_dir, f"{timestamp}_questions.json")
output_path_by_label = os.path.join(results_dir, f"{timestamp}_by_label.json")
output_path_by_combination = os.path.join(results_dir, f"{timestamp}_by_combination.json")
completed_file = os.path.join("data/combinations", "combinations_completed.json")
pending_file = os.path.join("data/combinations", "combinations_pending.json")
config_path = "configs/settings_ranges.yaml"
# Tipo de logging
auth_mode = "cookies"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluador de configuración actual del chatbot")
    parser.add_argument('--mode', choices=["combinations", "current"], default="combinations", help="Modo de ejecución: iterar combinaciones y obtener la mejor")
    return parser.parse_args()

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

@lru_cache(maxsize=1)
def load_metric_weights(weight_file_path: str = "configs/metric_weights.yaml") -> dict:
    logger.info(f"[INFO] Cargando pesos de métricas desde: {weight_file_path}")
    if not os.path.exists(weight_file_path):
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {weight_file_path}")
    with open(weight_file_path, 'r') as f:
        return yaml.safe_load(f)

def load_parameter_combinations(config_path):
    """
    Load parameter combinations for evaluation.
    Priority:
    1. If 'combinations_pending.json' exists, load from it.
    2. Otherwise, generate from 'settings_ranges.yaml'.

    Args:
        config_path (str): Path to the YAML config file with parameter ranges.
        results_dir (str): Directory where results and pending files are stored.

    Returns:
        List[dict]: List of parameter combinations to evaluate.
    """
    if os.path.exists(pending_file):
        logger.info(f"Cargando combinaciones pendientes desde archivo: {pending_file}")
        with open(pending_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        logger.info(f"Generando combinaciones de parámetros desde: {config_path}")
        num_samples = int(os.environ.get("NUM_SAMPLES", 4))  # Default to 4 if not set
        combinations = generate_parameter_combinations(config_path, num_samples)
        with open(pending_file, "w", encoding="utf-8") as f:
            json.dump(combinations, f, ensure_ascii=False)

        return combinations

def generate_parameter_combinations(config_path, num_samples=4):
    """
    Generate all parameter combinations from a YAML config file.
    Adds an 'id' field to each combination for identification.

    Args:
        config_path (str): Path to YAML file with parameter ranges.
        num_samples (int): Number of samples per ranged parameter.

    Returns:
        List[dict]: All parameter combinations with an 'id'.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config_ranges = yaml.safe_load(f)

    logger.info(f"Cargando combinaciones de parámetros desde archivo YAML: {config_path}")

    keys, values = [], []
    for key, val in config_ranges.items():
        keys.append(key)
        if isinstance(val, list):
            values.append(val)
        elif isinstance(val, dict):
            param_type, vmin, vmax = val.get("type"), val.get("min"), val.get("max")
            if param_type == "float":
                samples = np.linspace(vmin, vmax, num_samples).round(3).tolist() # type: ignore
            elif param_type == "int":
                samples = np.linspace(vmin, vmax, num_samples).astype(int).tolist() # type: ignore
            else:
                raise ValueError(f"[ERROR] Tipo desconocido '{param_type}' para parámetro '{key}'")
            values.append(samples)
        else:
            raise ValueError(f"[ERROR] Formato inesperado para parámetro '{key}'")

    raw_combinations = list(product(*values))
    combinations = [{"id": i + 1, **dict(zip(keys, v))} for i, v in enumerate(raw_combinations)]

    print(f"[INFO] Se generaron {len(combinations)} combinaciones de parámetros.")
    logger.info(f"Se generaron {len(combinations)} combinaciones de parámetros.")

    return combinations

def evaluate_combination(config):
    scores = []

    # 1. Evalua el grupo de preguntas para la configuración dada
    for label, questions in load_questions_by_label().items():
        config_score = evaluate_label(label, questions, config)
        scores.append(config_score)

    # 2. Calcular el promedio de todos los label_scores usando calculate_average
    combination_score = calculate_average(scores)
    
    result = {
    "config": config,
    "total_score": combination_score
    }

    # 3. Registrar el resultado completo

    with open(output_path_by_combination, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False)+ "\n")

    return config

def evaluate_label(label, questions, config):
    label_results = {}
    # 1. Evaluar cada grupo de preguntas por su etiqueta
    for question in questions:
        label_score = evaluate_question(question, config, label)
        label_results[label] = label_score

    # 2. Calcular el promedio de todos los label_scores usando calculate_average
    combination_score = calculate_average(list(label_results.values()))

    # 3. Crear el resultado completo con metadata
    result = {
        "config": config,
        "label_scores": label_results,
        "score_by_label": combination_score
    }

    with open(output_path_by_label, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False)+ "\n" )

    return combination_score

def evaluate_question(question, config, label):
    id = question['id']
    query = question['query']
    expected= question['expected_answer']

    #1. Realizar cada pregunta al chatbot

    response = try_send_question_with_retries(query)

    #2. Evaluar cada pregunta
    score = evaluate_response(
    qid=id,
    label=label,
    query=query,
    response=response['answer'], # type: ignore
    context=response['context'], # type: ignore
    ground_truth=expected)
    result = compute_weighted_score(score)

    #3. Guardar el score por pregunta
    score_entry = {
    "combination": config,
    "id": id,
    "label": label,
    "latency": response['latency'], # type: ignore
    "question": query,
    "expected": expected,
    "answer": response['answer'], # type: ignore
    "context": response['context'], # type: ignore
    "score": result
}
 
    # Guardar score por pregunta
    with open(output_path_questions, "a", encoding="utf-8") as f:
        f.write(json.dumps(score_entry,indent=2, ensure_ascii=False)+ "\n")

    logger.info(f"Pregunta {id} evaluada: {result["weighted_score"]} para la etiqueta '{label}'")
    return result["weighted_score"]

import time

def try_send_question_with_retries(query, max_retries=3, delay_seconds=5):
    for attempt in range(1, max_retries + 1):
        try:
            response = send_question_to_target(query)
            return response
        except RuntimeError as e:
            logger.warning(f"⚠️ Intento {attempt}/{max_retries} fallido: {e}")
            if attempt < max_retries:
                time.sleep(delay_seconds)
            else:
                logger.error(f"❌ Error tras {max_retries} intentos: {e}")
                raise  # <-- Propaga el error para que .main lo capture


def load_questions_by_label(file_path="data/generated_questions_by_tag.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def compute_weighted_score(score: dict) -> dict:
    weights_data = load_metric_weights()

    label = score.get("label")
    if not label:
        raise ValueError("El campo 'label' no está presente en el score")

    label_weights = weights_data.get("weights", {}).get(label)
    if not label_weights:
        raise ValueError(f"No se encontraron pesos para la etiqueta '{label}'")

    weighted_metrics = {}
    total = 0.0

    for metric_name, weight in label_weights.items():
        metric_value = score.get(metric_name)

        # Validar que el valor no sea None ni NaN
        if metric_value is not None and not (isinstance(metric_value, float) and math.isnan(metric_value)):
            weighted = metric_value * weight
            weighted_metrics[metric_name] = round(weighted, 4)
            total += weighted
        else:
            logger.warning(f"⚠️ La métrica '{metric_name}' es inválida (None o NaN) y será ignorada para la etiqueta '{label}'.")

    weighted_metrics["weighted_score"] = round(total, 4)
    return weighted_metrics

def calculate_average(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 2)

def main():
    start_time = time.time()
    elapsed = None #inicializar la variable elapsed para el reporte
    try:
        args = parse_args()
        reporter = EmailReporter() #se crea una instancia del reporter para enviar el reporte al finalizar o fallar
        elapsed = None #inicializar la variable elapsed para el reporte
        logger.info(f"Inicio del proceso en modo: {args.mode}")
        # Comprobar variables de entorno
        env_vars = check_env_vars(["SUBSCRIPTION_ID", "RESOURCE_GROUP", "WEB_APP_NAME", "WEBCHAT_URL"])

        if args.mode == "current":
            logger.info("Modo 'current' seleccionado. Actualizando configuración actual del Web App.")
            # 1. Cargar configuración actual del Web App
            current_settings = update_webapp_settings(
                subscription_id=env_vars["SUBSCRIPTION_ID"], 
                resource_group=env_vars["RESOURCE_GROUP"], 
                webapp_name=env_vars["WEB_APP_NAME"],
                mode= args.mode,
                new_settings={}
            )
            logger.info(f"Configuración actual del Web App: {current_settings}")
            result = evaluate_combination(current_settings)
        else:
            # Cargar combinaciones de parámetros
            combinations = load_parameter_combinations(config_path)
            combinations_pending = combinations  # Copiar para evitar modificar la lista original
            total_count = len(combinations)
            count=1
            for combination in combinations:
                print(f"Evaluando combinación {count}/{total_count}: {combination}")
                update_webapp_settings(
                    subscription_id=env_vars["SUBSCRIPTION_ID"], 
                    resource_group=env_vars["RESOURCE_GROUP"], 
                    webapp_name=env_vars["WEB_APP_NAME"],
                    mode= args.mode,
                    new_settings=combination
            )
                
                result = evaluate_combination(combination)

                # Añadir al archivo de completados (como línea JSON)
                with open(completed_file, "a", encoding="utf-8") as f_completed:
                    f_completed.write(json.dumps(combination) + "\n")

                # 4. Eliminar de la lista de pendientes
                combinations_pending = [c for c in combinations_pending if c["id"] != count]

                # 5. Sobrescribir el archivo de pendientes actualizado
                with open(pending_file, "w", encoding="utf-8") as f_pending:
                    json.dump(combinations_pending, f_pending)

                logger.info(f"Evaluación completada para la configuración: {combination}, resultado: {result}")
                count += 1
            
        #eliminar el archivo de completados si está vacío
        if os.path.exists(Path(pending_file)):
            os.remove(Path(pending_file))

        with open(output_path_by_combination, "r", encoding="utf-8") as f:
            bloques = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', f.read(), re.DOTALL)

        data = [json.loads(b) for b in bloques]

        with open(output_path_by_combination, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Enviar reporte por email
        end_time = time.time()
        elapsed = end_time - start_time
        reporter.send_report(args.mode,output_path_by_combination, elapsed)

    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        reporter.send_report("error", f"combinación: {count} - " + str(e), elapsed)
        logger.exception(f"Error en la ejecución principal: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()