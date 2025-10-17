"""
Logging utilities: colorized console and rotating file logs.
Always preserves a complete DEBUG log in logs/debug.log.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BLUE = "\033[94m"
C_RESET = "\033[0m"


def setup_logger(log_dir: str, console_level: int = logging.DEBUG) -> logging.Logger:
    """
    Create or return the global logger with color console and rotating files.
    The debug file always captures full DEBUG output regardless of console level.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")
    debug_path = os.path.join(log_dir, "debug.log")

    class ColorFormatter(logging.Formatter):
        def format(self, record):
            color = {logging.INFO: C_BLUE, logging.WARNING: C_YELLOW, logging.ERROR: C_RED}.get(record.levelno, "")
            msg = super().format(record)
            return f"{color}{msg}{C_RESET}"

    logger = logging.getLogger("llm_plotbot")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(console_level)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    dh = RotatingFileHandler(debug_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    dh.setLevel(logging.DEBUG)
    dh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(dh)

    logger.debug("Logger initialized at DEBUG level.")
    return logger
