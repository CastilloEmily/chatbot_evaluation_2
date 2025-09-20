import json
import smtplib
from email.message import EmailMessage
from tabulate import tabulate
from datetime import timedelta
import os
import logging
# Configuración del logger
logger = logging.getLogger("EmailReporter")

class EmailReporter:
    def __init__(self):
        logger.info("Inicializando EmailReporter y cargando variables de entorno...")

        self.smtp_server = os.environ['SMTP_SERVER']
        self.smtp_port = int(os.environ['SMTP_PORT'])
        self.email_user = os.environ['EMAIL_USER']
        self.email_pass = os.environ['EMAIL_PASS']
        self.email_to = os.environ['EMAIL_TO']

        logger.info("Variables de entorno cargadas correctamente.")

    def _load_json_data(self, file_path):
        logger.info(f"Cargando archivo JSON desde: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Archivo JSON cargado correctamente. Registros totales: {len(data)}")
                return data
        except Exception as e:
            logger.error(f"Error al cargar el archivo JSON: {e}")
            raise e

    def _generate_html_table(self, top_10):
        logger.info("Generando tabla HTML para el top 10 de configuraciones.")
        dynamic_keys = list(top_10[0]["config"].keys())
        table_headers = ["Rank", "Score"] + [key.replace("AZURE_", "").replace("_", " ").title() for key in dynamic_keys]
        table_data = []

        for i, item in enumerate(top_10, 1):
            cfg = item["config"]
            row = [i, round(item["total_score"], 3)] + [cfg.get(key, "") for key in dynamic_keys]
            table_data.append(row)

        logger.info("Tabla HTML generada correctamente.")
        return tabulate(table_data, headers=table_headers, tablefmt="html"), table_data, table_headers

    def _style_html_table(self, table_html):
        logger.debug("Aplicando estilos a la tabla HTML.")
        html_style = """
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            th {
                background-color: #FF8708;
                color: white;
            }
            tr:nth-child(even){background-color: #f2f2f2;}
            tr:hover {background-color: #ddd;}
        </style>
        """
        return html_style + table_html

    def generate_current_config_table(self, config_dict: dict) -> str:
        rows = ""
        for key, value in sorted(config_dict.items()):
            value_display = value if value else "<i>(vacío)</i>"
            rows += f"<tr><td><b>{key}</b></td><td>{value_display}</td></tr>\n"
        
        return f"""
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; font-family: Arial; font-size: 14px;">
            <thead style="background-color: #f2f2f2;">
                <tr><th>Parámetro</th><th>Valor</th></tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """

    def send_report(self, mode: str, file_path: str, execution_time: float):
        logger.info("Iniciando envío de reporte por correo.")
        formatted_time = str(timedelta(seconds=int(execution_time)))

        if mode == "error":
            email_subject = "❌ Error en la ejecución del Script Evaluador"
            intro_message = "Se ha producido un error durante la ejecución del script."
            summary_html = f"""
                <p><b>🧾 Detalles del error:</b></p>
                <pre style="background-color:#f8d7da; color:#721c24; padding:10px; border-radius:5px;">
                {file_path}
                </pre>
                <p>Por favor, revisa el script o los registros de ejecución para más información.</p>
            """

        else:
            # Solo procesar datos si no es un modo de error
            data = self._load_json_data(file_path)
            sorted_data = sorted(data, key=lambda x: x['total_score'], reverse=True)
            top_10 = sorted_data[:10]

            best_config = top_10[0]
            logger.info(f"Mejor configuración encontrada con score: {best_config['total_score']}")

            table_html_raw, _, _ = self._generate_html_table(top_10)
            styled_table_html = self._style_html_table(table_html_raw)

            if mode == "current":
                email_subject = "📌 Score de la Configuración Actual"
                intro_message = "Aquí tienes el resultado de la evaluación de la configuración actual."

                config_table_html = self.generate_current_config_table(best_config["config"])
                summary_html = f"""
                    <p><b>🏆 Score combinación actual:</b> {best_config['total_score']}</p>
                    <p><b>📋 Configuración actual:</b></p>
                    {config_table_html}
                """

            elif mode == "combinations":
                email_subject = "📊 Ranking de Configuraciones - Top 10 por Total Score"
                intro_message = "Aquí tienes el ranking de las 10 mejores configuraciones."
                summary_html = f"""
                    <p><b>🏆 Mejor configuración:</b> {best_config["config"]["id"]} con Score: {best_config['total_score']}</p>
                    <p><b>📋 Tabla resumen:</b></p>
                    {styled_table_html}
                """

        # Email HTML común a todos los modos
        email_html = f"""
        <html>
        <head></head>
        <body>
            <p>Hola,</p>
            <p>{intro_message}</p>
            <p><b>⏱ Tiempo total de ejecución del script:</b> {formatted_time}</p>
            {summary_html}
            <p>Saludos,<br>Tu Script Evaluador</p>
        </body>
        </html>
        """

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_pass)

                msg = EmailMessage()
                msg["Subject"] = email_subject
                msg["From"] = self.email_user
                msg["To"] = self.email_to
                msg.set_content("Este mensaje requiere un cliente compatible con HTML.")
                msg.add_alternative(email_html, subtype="html")
                server.send_message(msg)

            logger.info("✅ Correo enviado con éxito.")
        except Exception as e:
            logger.error(f"❌ Error al enviar el correo: {e}")
            raise e

#if __name__ == "__main__":
#    reporter = EmailReporter()
#    reporter.send_report("C:/Users/CASTILLOEMILYGABRIEL/OneDrive - TK Elevator/Documentos/chatbot_evaluation/results/2025-05-29_16-23-16/2025-05-29_16-23-16_by_combination.json", float(3.14))