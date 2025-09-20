# utils/logger.py
import logging
import os
from datetime import datetime

def setup_logging():
    now = datetime.now()
    day_str = now.strftime("%Y-%m-%d")
    full_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = os.path.join("logs", day_str)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{full_str}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("Logging configurado. Archivo de log: %s", log_file)
