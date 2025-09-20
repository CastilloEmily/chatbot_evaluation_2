import typer
from pathlib import Path
from generate_v2 import (
    generate_test_qa_data_for_search_index,
    get_openai_config_dict,
    get_search_client,
    generate_by_tag as generate_by_tag_impl,
)

app = typer.Typer()

@app.command()
def generate(
    output: Path = typer.Option(..., exists=False, dir_okay=False, file_okay=True),
    numquestions: int = typer.Option(200, help="Number of questions to generate"),
    persource: int = typer.Option(5, help="Number of questions to generate per source"),
    citationfieldname: str = typer.Option("filepath", help="Name of citation field in AI search index"),
):
    generate_test_qa_data_for_search_index(
        openai_config=get_openai_config_dict(),
        search_client=get_search_client(),
        num_questions_total=numquestions,
        num_questions_per_source=persource,
        output_file=Path.cwd() / output,
        citation_field_name=citationfieldname,
    )

@app.command(name="generate-by-tag")
def generate_by_tag(
    tagpromptfile: Path = typer.Option(..., exists=True, help="Archivo JSON con las etiquetas y sus prompts"),
    output: Path = typer.Option(..., exists=False, dir_okay=False, file_okay=True, help="Archivo de salida JSON"),
    numquestions: int = typer.Option(..., help="Número total de preguntas a generar"),
    persource: int = typer.Option(5, help="Número de preguntas por fuente"),
    citationfieldname: str = typer.Option("filepath", help="Campo de cita en el índice de búsqueda AI"),
):
    return generate_by_tag_impl(
        tagpromptfile=tagpromptfile,
        output=output,
        numquestions=numquestions,
        persource=persource,
        citationfieldname=citationfieldname,
    )

if __name__ == "__main__":
    app()