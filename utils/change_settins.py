import logging
import time
import sys
import traceback
from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
from azure.core.exceptions import HttpResponseError

# Configuración avanzada de logging
logger = logging.getLogger("change_settings")
#logger.setLevel(logging.DEBUG)  # Loguear desde DEBUG en adelante


def update_webapp_settings(
    subscription_id: str,
    resource_group: str,
    webapp_name: str,
    new_settings: dict,
    mode: str,
    wait_time: int = 5,
    max_retries: int = 10,
):
    logger.info("Inicio actualización settings de Web App '%s' en RG '%s', en subscription '%s'", webapp_name, resource_group, subscription_id)
    try:
        credential = DefaultAzureCredential()
        client = WebSiteManagementClient(credential, subscription_id)

        # Obtener configuración actual
        current_config = client.web_apps.list_application_settings(resource_group, webapp_name)
        current_settings = current_config.properties or {}
        logger.info("Configuración actual obtenida correctamente.")

                # Si el modo es "current", devolver configuración actual sin cambios
        if mode == "current":
            logger.info("Modo 'current' activado. No se aplicarán cambios.")
            return  current_settings

        else:
        
            # Actualizar configuración
            settings = new_settings.copy()
            settings.pop("id", None)
            updated_settings = current_settings.copy()
            updated_settings.update(settings)

            client.web_apps.update_application_settings(
                resource_group, webapp_name, {"properties": updated_settings} # type: ignore
            ) # type: ignore
            logger.info("Nueva configuración aplicada correctamente.")

            # Reiniciar Web App
            client.web_apps.restart(resource_group, webapp_name)
            logger.info("Solicitud de reinicio enviada.")

            # Esperar a que esté 'Running'
            logger.info("Esperando a que la Web App esté en estado 'Running' (máximo %d intentos)...", max_retries)
            time.sleep(wait_time)
            for attempt in range(max_retries):
                status = client.web_apps.get(resource_group, webapp_name).state or "" # type: ignore
                logger.debug("Estado actual de la Web App: '%s' (Intento %d/%d)", status, attempt + 1, max_retries)
                if status.lower() == "running":
                    logger.info("La Web App está en estado 'Running'.")
                    break
                logger.warning("Estado '%s'. Reintentando en %d segundos...", status, wait_time)
                time.sleep(wait_time)
            else:
                err_msg = "La Web App no volvió a estado 'Running' tras el reinicio."
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            # Verificar configuración aplicada
            verified_config = client.web_apps.list_application_settings(resource_group, webapp_name)

            for key, expected_value in settings.items():
                actual_value = verified_config.properties.get(key)  # type: ignore

                # Normalizar ambos valores
                expected_str = str(expected_value).strip()
                actual_str = str(actual_value).strip()

                if expected_str != actual_str:
                    err_msg = (
                        f"Setting '{key}' no actualizado correctamente. "
                        f"Esperado: '{expected_str}', Obtenido: '{actual_str}'"
                    )
                    logger.error(err_msg)
                    raise ValueError(err_msg)

            logger.debug(f"Comparando clave {key}: repr(esperado)={repr(expected_str)} vs repr(actual)={repr(actual_str)}")

            return {
                "success": True,
                "message": "Configuración aplicada y Web App reiniciada correctamente."
            }

    except HttpResponseError as e:
        logger.error("Error HTTP en Azure API: %s", str(e))
        #logger.debug(traceback.format_exc())
        raise e  # Deja que .main lo capture y lo reporte por email

    except (ValueError, RuntimeError) as e:
        logger.error("Error de lógica: %s", str(e))
        #logger.debug(traceback.format_exc())
        raise e  # Deja que .main lo capture y lo reporte por email

    except Exception as e:
        logger.error("Error inesperado: %s", str(e))
        #logger.debug(traceback.format_exc())
        raise e  # Deja que .main lo capture y lo reporte por email

""""
if __name__ == "__main__":
    resultado = update_webapp_settings(
        subscription_id="b29c9dac-08df-4427-9ddb-d50fca7bc22f",
        resource_group="openAI",
        webapp_name="ChatRic",
        new_settings={
            "APP_MODE": "testing settings",
            "AZURE_OPENAI_KEY": "100"
        }
    )

    if resultado["success"]:
        logger.info("✅ %s", resultado["message"])
        print("✅", resultado["message"])
    else:
        logger.error("❌ %s", resultado["message"])
        print("❌", resultado["message"])
        sys.exit(1)
"""