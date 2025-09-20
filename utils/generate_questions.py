import os
import json
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.evaluation import QAGenerator
from azure.identity import DefaultAzureCredential

# === CONFIGURACIÓN ===
SEARCH_ENDPOINT = "https://<tu-servicio-search>.search.windows.net"
SEARCH_API_KEY = "<tu-clave-search>"
SEARCH_INDEX_NAME = "<nombre-del-indice>"

OPENAI_ENDPOINT = "https://<tu-recurso-openai>.openai.azure.com"
OPENAI_API_KEY = "<tu-clave-openai>"
OPENAI_DEPLOYMENT_NAME = "<nombre-del-deployment-gpt>"
OPENAI_API_VERSION = "2024-02-15-preview"  # actualiza si es necesario

OUTPUT_FILE = "preguntas_generadas.json"
NUM_DOCUMENTS = 10  # número de documentos a recuperar para generar preguntas

# === CLIENTE DE SEARCH ===
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

# === RECUPERACIÓN DE DOCUMENTOS ===
print("Recuperando documentos...")
results = search_client.search(search_text="*", top=NUM_DOCUMENTS)

documents = []
for result in results:
    content = result.get("content") or result.get("text") or json.dumps(result)
    documents.append(content)

# === GENERACIÓN DE PREGUNTAS ===
print("Generando preguntas...")

qa_generator = QAGenerator.from_openai(
    openai_endpoint=OPENAI_ENDPOINT,
    openai_api_key=OPENAI_API_KEY,
    deployment_name=OPENAI_DEPLOYMENT_NAME,
    api_version=OPENAI_API_VERSION,
)

all_questions = []
for i, doc in enumerate(documents):
    try:
        questions = qa_generator.run(input=doc)
        all_questions.append({
            "document_id": i,
            "content": doc[:200] + "...",  # Preview
            "questions": questions,
        })
        print(f"Preguntas generadas para documento {i}: {len(questions)}")
    except Exception as e:
        print(f"Error generando preguntas para documento {i}: {e}")

# === GUARDAR RESULTADOS ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_questions, f, indent=2, ensure_ascii=False)

print(f"Preguntas guardadas en {OUTPUT_FILE}")
