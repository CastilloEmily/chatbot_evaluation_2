import os
import json
import logging
import yaml
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
# Cargar variables de entorno desde .env
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', '.env')))
from utils.loader import response_generator
from app.evaluator import evaluate_response
from utils.change_settins import update_webapp_settings

# Obtener timestamp actual
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Crear carpetas de logs y resultados con timestamp
log_dir = os.path.join("logs", timestamp)
results_dir = os.path.join("results", timestamp)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Configurar logging con archivo en carpeta correspondiente
log_file_path = os.path.join(log_dir, f"log_{timestamp}.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    print(f"Metrica {nombre} con valor {valor}")
    if nombre == "f1_score":
        return valor
    elif nombre.startswith("gpt_") or nombre in [
        "similarity", "relevance", "groundedness", "coherence", "fluency"
    ]:
        return max(0.0, min((valor - 1) / 4, 1.0))
    return None

def main():
    try:
        logging.info("Inicio del proceso de evaluación del chatbot.")

        with open("data/questions.json", "r", encoding="utf-8") as f:
            questions_by_label = json.load(f)

        with open("configs/metric_weights.yaml", "r", encoding="utf-8") as f:
            metric_weights_by_label = yaml.safe_load(f)["weights"]

        auth_mode = "cookies"
        all_scores = []
        label_scores = defaultdict(list)

        for qid, label, question, expected, actual, context in response_generator(questions_by_label, auth_mode):
            print(f"Evaluando pregunta {question}")
            print(f"Respuesta esperada: {expected}")
            print(f"Respuesta obtenida: {actual}")
            if actual is None:
                logging.error(f"No se recibió respuesta para la pregunta {qid}")
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

            logging.info(f"Evaluación completada para la pregunta {qid}.")

        file_prefix = f"{timestamp}"

        # 1. Score por pregunta
        with open(os.path.join(results_dir, f"{file_prefix}_per_question_scores.json"), "w", encoding="utf-8") as f:
            json.dump(all_scores, f, indent=2, ensure_ascii=False)

        # 2. Score por etiqueta
        scores_por_grupo = defaultdict(list)
        for entry in all_scores:
            label = entry["label"]
            weighted = entry.get("weighted_score")
            if weighted is not None:
                scores_por_grupo[label].append(weighted)

        promedios_por_grupo = {
            label: round(sum(valores) / len(valores), 4)
            for label, valores in scores_por_grupo.items()
        }

        with open(os.path.join(results_dir, f"{file_prefix}_scores_by_label.json"), "w", encoding="utf-8") as f:
            json.dump(promedios_por_grupo, f, indent=2, ensure_ascii=False)

        # 3. Score global de métricas crudas
        all_raw_scores = [entry["score"] for entry in all_scores]
        global_average = calcular_promedio_metricas(all_raw_scores)

        # 4. Score global normalizado unificado (aplicando normalización solo al resultado final)
        score_unico = calcular_score_global_unificado(all_scores)
        score_normalizado = normalizar_metrica("weighted_global_score", score_unico) if score_unico is not None else 0.0
        global_average["normalized_global_score"] = score_normalizado if score_normalizado is not None else 0.0
        
        # 5. Score global ponderado (real)
        todos_los_ponderados = [entry["weighted_score"] for entry in all_scores if entry.get("weighted_score") is not None]
        score_global = round(sum(todos_los_ponderados) / len(todos_los_ponderados), 4)
        global_average["weighted_global_score"] = score_global

        # Guardar resumen final
        with open(os.path.join(results_dir, f"{file_prefix}_summary_scores.json"), "w", encoding="utf-8") as f:
            json.dump(global_average, f, indent=2, ensure_ascii=False)

        logging.info("Evaluación finalizada exitosamente.")
        print(f"✅ Evaluación completada. Revisa resultados en: {results_dir}")

    except Exception as e:
        logging.exception(f"Error en la ejecución principal: {str(e)}")
        print(f"❌ Error durante la evaluación. Ver detalles en: {log_file_path}")

if __name__ == "__main__":
    main()