          Ejecución del script de evaluación 

        Modo combinations (default si no pasas nada)

python -m app.main --mode combinations


        Modo combinations (default si no pasas nada)
        Carga combinaciones desde:
        data/combinations/combinations_pending.json si existe; o
        las genera a partir de configs/settings_ranges.yaml (usando NUM_SAMPLES si está en .env) y guarda ese listado en combinations_pending.json.

        Para cada combinación:
        Llama a update_webapp_settings(..., mode="combinations", new_settings=combination) para aplicar la configuración en el Web App.
        Lanza la evaluación (preguntas por etiqueta → puntuación por etiqueta → promedio total).
        Añade la combinación al archivo data/combinations/combinations_completed.json (append, una línea por combinación).
        Quita esa combinación del archivo combinations_pending.json (lo reescribe sin esa id).
        Al final, si combinations_pending.json quedó vacío, lo elimina.

        Salidas:
        results/<timestamp>/...
        <timestamp>_questions.json → una línea por pregunta evaluada (NDJSON).
        <timestamp>_by_label.json → una línea por etiqueta evaluada (NDJSON).
        <timestamp>_by_combination.json → primero NDJSON, luego lo reempaqueta a lista JSON.
        Reporte por email: EmailReporter().send_report("combinations", path_resultados, elapsed).

        Modo current

python -m app.main --mode current

        No toca pending ni completed.
        Llama a update_webapp_settings(..., mode="current", new_settings={}) para leer (o sincronizar) la config actual del Web App.
        Evalúa esa configuración y genera resultados en results/<timestamp>/....
        Reporte por email: EmailReporter().send_report("current", path_resultados, elapsed).

    Generar preguntas por tag ejemplo:

python .\cli.py generate-by-tag `
--tagpromptfile ..\configs\custom_prompts.json `
--output qa_por_etiquetas.json `
--numquestions 100 `
--persource 1 `
--citationfieldname filepath

    Generar preguntas normal:

python .\cli.py `
--output qa.json `
--numquestions 40 `
--persource 2 `
--citationfieldname filepath
