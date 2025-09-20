import os
import json
import logging
from dotenv import load_dotenv
# Cargar variables de entorno desde .env
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', '.env')))
from utils.loader import response_generator
from app.evaluator import evaluate_response
#from utils.report import save_results_summary

# Configuración de logs
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    try:
        logging.info("Inicio del proceso de evaluación del chatbot.")


        # Cargar preguntas
        with open("data/questions.json", "r", encoding="utf-8") as f:
            questions_by_label = json.load(f)

        auth_mode ="cookies"  # cookies o api_key

        all_scores = []

        for qid, label, question, expected, actual, context in response_generator(questions_by_label, auth_mode):
            print(f"Evaluando pregunta {qid} con etiqueta {label}...")
            if actual is None:
                logging.error(f"No se recibió respuesta para la pregunta {qid}")
                continue
 
            score = evaluate_response(
                qid=qid,
                label=label,
                query=question,
                response=expected,
                context=context,
                ground_truth=expected)

            score_entry = {
                "id": qid,
                "label": label,
                "question": question,
                "expected": expected,
                "actual": actual,
                "context": context,
                "score": score
            }
            all_scores.append(score_entry)

            logging.info(f"Evaluación completada para la pregunta {qid} con puntaje {score}")

        # Guardar resultados
        os.makedirs("results", exist_ok=True)
        summary_file = "results/summary_scores.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, indent=2, ensure_ascii=False)

        # Generar informe si aplica
        #save_results_summary(all_scores, output_path="results/summary_report.txt")

        logging.info("Proceso de evaluación finalizado exitosamente.")

    except Exception as e:
        logging.exception(f"Error en la ejecución principal: {str(e)}")

if __name__ == "__main__":
    main()


