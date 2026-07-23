import logging
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

logger = logging.getLogger("VAE_Auditor")

if not logger.handlers:

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(message)s"
    )

    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def _write(level: str, event: str, details: str = ""):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = (
        f"[{timestamp}] "
        f"[{level}] "
        f"[{event}] "
        f"{details}"
    )

    if level == "INFO":
        logger.info(message)

    elif level == "WARNING":
        logger.warning(message)

    else:
        logger.error(message)


def log_info(event, details=""):
    _write("INFO", event, details)


def log_warning(event, details=""):
    _write("WARNING", event, details)


def log_error(event, details=""):
    _write("ERROR", event, details)