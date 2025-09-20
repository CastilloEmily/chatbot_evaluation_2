import os
import json
import logging
import math
import random
from collections.abc import Generator
from pathlib import Path
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import openai
import typer
from typing import Callable, Optional
import sys
from urllib.parse import urlparse  # 👈 nuevo

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', '.env')))
logger = logging.getLogger("generate")
app = typer.Typer()

# Utilidad: colapsa saltos de línea y espacios
def to_oneline(text: str) -> str:
    return " ".join((text or "").split())

# 👇 Utilidad: obtener nombre de documento desde ruta o URL
def extract_doc_name(raw: str) -> str:
    if not raw:
        return "this document"
    p = urlparse(str(raw))
    path = p.path if p.scheme else str(raw)
    path = path.replace("\\", "/")
    name = path.split("/")[-1] if "/" in path else path
    return name or "this document"

@app.command()
def generate(
    output: Path = typer.Option(exists=False, dir_okay=False, file_okay=True),
    numquestions: int = typer.Option(help="Number of questions to generate", default=200),
    persource: int = typer.Option(help="Number of questions to generate per source", default=5),
    citationfieldname: str = typer.Option(help="Name of citiation field in ai search index", default="filepath"),
):
    generate_test_qa_data_for_search_index(
        openai_config=get_openai_config_dict(),
        search_client=get_search_client(),
        num_questions_total=numquestions,
        num_questions_per_source=persource,
        output_file=Path.cwd() / output,
        citation_field_name=citationfieldname,
    )

@app.command()
def generate_dontknows(
    input: Path = typer.Option(exists=True, dir_okay=False, file_okay=True),
    output: Path = typer.Option(exists=False, dir_okay=False, file_okay=True),
    numquestions: int = typer.Option(help="Number of questions to generate", default=40),
):
    generate_dontknows_qa_data(
        openai_config=get_openai_config(),
        num_questions_total=numquestions,
        input_file=Path.cwd() / input,
        output_file=Path.cwd() / output,
    )

def get_openai_config_dict():
    openai_config = {
        "api_type": "azure",
        "api_base": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
        "deployment": os.environ["MODEL_DEPLOYMENT_NAME"],
        "model": os.environ["MODEL_NAME"],
    }
    return openai_config

def get_openai_config():
    openai_config = {
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "azure_deployment": os.environ["MODEL_DEPLOYMENT_NAME"],
        "model": os.environ["MODEL_NAME"],
    }
    return openai_config

def get_openai_client():
    return openai.AzureOpenAI(
        api_version="2024-02-15-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["MODEL_DEPLOYMENT_NAME"],
    )

def get_search_client():
    api_key = os.environ.get("AZURE_SEARCH_KEY")
    if not api_key:
        raise ValueError("AZURE_SEARCH_KEY environment variable is not set.")
    credentials = AzureKeyCredential(api_key)
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX"],
        credential=credentials,
    )

def build_prompt(tag_prompt: str, source_text: str, num_questions: int, doc_label: str) -> str:
    return (
        f"{tag_prompt}\n\n"
        f"Document identifier (use this exact string in every question): {doc_label}\n\n"
        f"Content:\n{source_text}\n\n"
        f"Generate {num_questions} question–answer pairs strictly based on this content.\n"
        f"Requirements:\n"
        f"- Each question must explicitly reference the document by name (e.g., start with \"In '{doc_label}', ...\" or \"According to '{doc_label}', ...\").\n"
        f"- Keep questions specific and answers concise.\n"
        f"- Do not invent facts; only use the provided content.\n"
        f"Format:\nQ: ...\nA: ...\n\nSeparate each pair with a blank line."
    )

def generate_test_qa_data(
    openai_config: dict,
    num_questions_total: int,
    num_questions_per_source: int,
    output_file: Path,
    source_retriever: Callable[[], Generator[dict, None, None]],
    source_to_text: Callable,
    answer_formatter: Callable,
    prompt: Optional[str] = None,
) -> Optional[list[dict]]:
    client = get_openai_client()
    logger.info(
        "Generating %d questions total, %d per source, using ChatGPT model",
        num_questions_total,
        num_questions_per_source,
    )

    qa: list[dict] = []
    sources = list(source_retriever())
    num_source = len(sources)

    # cuántas FUENTES necesito si cada fuente genera 'num_questions_per_source' pares
    sources_needed = max(1, math.ceil(num_questions_total / max(1, num_questions_per_source)))

    if num_source == 0:
        logger.warning("No hay fuentes disponibles.")
        selected_sources = []
    else:
        take = min(num_source, sources_needed)
        selected_sources = random.sample(sources, take)

    for source in selected_sources:
        if len(qa) >= num_questions_total:
            logger.info("Generated enough questions already, stopping")
            break

        source_text = source_to_text(source)
        doc_label = str(
            source.get("filepath")
            or source.get("title")
            or source.get("document")
            or source.get("id")
            or "this document"
        )
        doc_name = extract_doc_name(doc_label)  # 👈 nombre corto del documento
        # 👇 ground_truth con nombre del documento antes del chunk
        ground_truth_oneline = f"[{doc_name}] {to_oneline(source_text)}"

        final_prompt = (
            f"You are given an excerpt from a single source document.\n"
            f"Document identifier (use this exact string in every question): {doc_label}\n\n"
            f"Content:\n{source_text}\n\n"
            f"Generate {num_questions_per_source} question–answer pairs strictly based on the content above.\n"
            f"Requirements:\n"
            f"- Each question must explicitly reference the document by name (e.g., start with \"In '{doc_label}', ...\" or \"According to '{doc_label}', ...\").\n"
            f"- Do not use external knowledge; rely only on the provided content.\n"
            f"Format:\nQ: ...\nA: ...\n\nSeparate each pair with a blank line."
        ) if prompt is None else build_prompt(prompt, source_text, num_questions_per_source, doc_label)

        try:
            response = client.chat.completions.create(
                model=openai_config["model"],
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.3,
                max_tokens=1000,
            )

            content = response.choices[0].message.content.strip()  # type: ignore
            pairs = content.split("\n\n")

            parsed_pairs = []
            for pair in pairs:
                if "Q:" in pair and "A:" in pair:
                    try:
                        question = pair.split("Q:")[1].split("A:")[0].strip()
                        answer = pair.split("A:")[1].strip()
                        parsed_pairs.append({
                            "query": question,
                            "truth": answer_formatter(answer, source),
                            "ground_truth": ground_truth_oneline,  # 👈 con nombre del doc
                        })
                    except Exception as e:
                        logger.warning("Failed to parse pair: %s", e)

            # Acumulamos las preguntas generadas
            qa.extend(parsed_pairs)

        except Exception as e:
            logger.warning("OpenAI API call failed: %s", e)

    # Si se pasó prompt → devolver lista; si no, escribir JSONL
    if prompt is not None:
        return qa
    else:
        logger.info("Writing %d questions to %s", len(qa), output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in qa[:num_questions_total]:
                f.write(json.dumps(item) + "\n")
        return None

def source_retriever_unique_docs(search_client: SearchClient, page_size: int = 1000) -> list[dict]:
    # Trae TODO el índice paginando (no solo 1000)
    try:
        total = search_client.get_document_count()
    except Exception:
        total = 100000  # fallback

    docs_by_file: dict[str, list[dict]] = {}
    skip = 0
    while skip < total:
        batch = list(search_client.search("", top=page_size, skip=skip))
        if not batch:
            break
        for doc in batch:
            fp = str(doc.get("filepath") or doc.get("document") or doc.get("id") or "UNKNOWN")
            docs_by_file.setdefault(fp, []).append(doc)
        skip += page_size

    # Devuelve una lista “equilibrada”: 1–2 chunks por doc
    balanced_sources: list[dict] = []
    for fp, chunks in docs_by_file.items():
        take = min(2, len(chunks))
        balanced_sources.extend(random.sample(chunks, take))
    return balanced_sources

def generate_test_qa_data_for_search_index(
    openai_config: dict,
    num_questions_total: int,
    num_questions_per_source: int,
    output_file: Path,
    search_client: SearchClient,
    citation_field_name: str,
    prompt: Optional[str] = None,
):
    # cuántas FUENTES necesitas si cada fuente genera 'num_questions_per_source' pares
    sources_needed = max(1, math.ceil(num_questions_total / max(1, num_questions_per_source)))
    margin_factor = 2  # pequeño margen para tener de dónde elegir

    def source_retriever() -> Generator[dict, None, None]:
        # Obtén una lista balanceada por documento
        sources = source_retriever_unique_docs(search_client)
        if not sources:
            return  # termina el generador si no hay fuentes

        # Limita la cantidad de fuentes a consultar para acelerar
        k = min(len(sources), sources_needed * margin_factor)
        for s in random.sample(sources, k):
            yield s

    def source_to_text(source) -> str:
        return source["text"]

    def answer_formatter(answer, source) -> str:
        return answer  # 👈 sin URL ni corchetes

    qa = generate_test_qa_data(
        openai_config=openai_config,
        num_questions_total=num_questions_total,
        num_questions_per_source=num_questions_per_source,
        output_file=output_file,
        source_retriever=source_retriever,
        source_to_text=source_to_text,
        answer_formatter=answer_formatter,
        prompt=prompt,
    )
    return qa

def generate_based_on_questions(openai_client, model: str, qa: list, num_questions: int, prompt: str):
    existing_questions = ""
    if qa:
        qa = random.sample(qa, len(qa))  # Shuffle questions for some randomness
        existing_questions = "\n".join([item["query"] for item in qa])

    gpt_response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"{prompt} Only generate {num_questions} TOTAL. Separate each question by a new line. \n{existing_questions}",  # noqa: E501
            }
        ],
        n=1,
        max_tokens=num_questions * 50,
        temperature=0.3,
    )

    qa = []
    for message in gpt_response.choices[0].message.content.split("\n")[0:num_questions]:
        qa.append({"query": message, "truth": f"Generated from this prompt: {prompt}"})
    return qa

def generate_dontknows_qa_data(openai_config: dict, num_questions_total: int, input_file: Path, output_file: Path):
    logger.info("Generating off-topic questions based on %s", input_file)
    openai_config = get_openai_config()
    with open(input_file, encoding="utf-8") as f:
        qa = [json.loads(line) for line in f.readlines()]

    openai_client = get_openai_client()
    dontknows_qa = []
    num_questions_each = math.ceil(num_questions_total / 4)
    dontknows_qa += generate_based_on_questions(
        openai_client,
        openai_config["model"],
        qa,
        num_questions_each,
        f"Given these questions, suggest {num_questions_each} questions that are very related but are not directly answerable by the same sources. Do not simply ask for other examples of the same thing - your question should be standalone.",  # noqa: E501
    )
    dontknows_qa += generate_based_on_questions(
        openai_client,
        openai_config["model"],
        qa,
        num_questions_each,
        f"Given these questions, suggest {num_questions_each} questions with similar keywords that are about publicly known facts.",  # noqa: E501
    )
    dontknows_qa += generate_based_on_questions(
        openai_client,
        openai_config["model"],
        qa,
        num_questions_each,
        f"Given these questions, suggest {num_questions_each} questions that are not related to these topics at all but have well known answers.",  # noqa: E501
    )
    remaining = num_questions_total - len(dontknows_qa)
    dontknows_qa += generate_based_on_questions(
        openai_client,
        openai_config["model"],
        qa=None,  # type: ignore
        num_questions=remaining,
        prompt=f"Suggest {remaining} questions that are nonsensical, and would result in confusion if you asked it.",  # noqa: E501
    )

    logger.info("Writing %d off-topic questions to %s", len(dontknows_qa), output_file)
    directory = Path(output_file).parent
    if not directory.exists():
        directory.mkdir(parents=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dontknows_qa:
            f.write(json.dumps(item) + "\n")

def generate_by_tag(
    tagpromptfile: Path = typer.Option(..., exists=True, help="Archivo JSON con las etiquetas y sus prompts"),
    output: Path = typer.Option(..., exists=False, dir_okay=False, file_okay=True, help="Archivo de salida JSON"),
    numquestions: int = typer.Option(..., help="Número total de preguntas a generar"),
    persource: int = typer.Option(default=5, help="Número de preguntas por fuente (requerido internamente)"),
    citationfieldname: str = typer.Option(default="filepath", help="Campo de cita en el índice de búsqueda AI")
):
    """
    Genera preguntas clasificadas por etiquetas, usando prompts personalizados cargados desde un archivo JSON.
    """
    with open(tagpromptfile, "r", encoding="utf-8") as f:
        tag_prompt_dict = json.load(f)

    tags = list(tag_prompt_dict.keys())
    num_tags = len(tags)
    questions_per_tag = numquestions // num_tags
    remainder = numquestions % num_tags

    output_data = {}
    current_question_id = 1

    for idx, tag in enumerate(tags):
        prompt = tag_prompt_dict[tag]
        count = questions_per_tag + (1 if idx < remainder else 0)
        sources_needed = max(1, math.ceil(count / persource))

        # Reutilizamos la función ya existente de generación
        qa = generate_test_qa_data_for_search_index(
            openai_config=get_openai_config_dict(),
            search_client=get_search_client(),
            num_questions_total=sources_needed,
            num_questions_per_source=persource,
            output_file=Path(f"temp_{tag}.json"),
            citation_field_name=citationfieldname,
            prompt=prompt
        )

        if not qa:
            logger.warning("No se generaron preguntas para la etiqueta '%s'. Continúo.", tag)
            continue

        k = min(count, len(qa))
        if k < count:
            logger.warning(
                "La etiqueta '%s' generó solo %d de %d preguntas solicitadas. Usando todas las disponibles.",
                tag, len(qa), count
            )
        selected_questions = random.sample(qa, k)

        tag_questions = []
        for q in selected_questions:
            tag_questions.append({
                "id": f"{tag[:2]}{current_question_id}",
                "query": q["query"],
                "expected_answer": q["truth"],
                "ground_truth": q.get("ground_truth", ""),  # ya incluye [docname] al inicio
            })
            current_question_id += 1

        output_data[tag] = tag_questions

    # Escribir UNA sola vez al final
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)