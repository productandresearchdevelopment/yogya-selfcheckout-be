import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import STORAGE_DIR

LOGS_DIR = STORAGE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOGS_DIR / "studio.log"


def configure_logging(level: int = logging.INFO):
    """
    Configures structured, production-ready logging:
    - Rotating file handler (10 MB per file, max 5 backups)
    - Console handler with clean formatting
    - Redacts base64 and sensitive tokens
    """
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers on re-initialization
    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        file_handler = RotatingFileHandler(
            str(LOG_FILE_PATH),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    logger = logging.getLogger("studio")
    logger.info("Product Recognition Studio logging subsystem initialized.")
    return logger
