"""Centralised logging configuration helpers."""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Mapping


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, *, use_color: bool) -> None:
        super().__init__(fmt)
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - colour branch
        message = super().format(record)
        if not self.use_color:
            return message
        color = self.COLORS.get(record.levelname)
        if not color:
            return message
        return f"{color}{message}{self.RESET}"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - logging
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(config: Mapping[str, object]) -> logging.Logger:
    """Configure console and file loggers according to configuration."""

    console_level = _coerce_level(config.get("console_level"))
    file_level = _coerce_level(config.get("file_level"))
    json_logs = bool(config.get("json_logs"))
    use_color = bool(config.get("color", True))
    log_dir = config.get("log_dir") or config.get("logs") or config.get("directory")

    logger = logging.getLogger("LLMPlotBot")
    logger.setLevel(logging.DEBUG)

    # Remove previous handlers to avoid duplicates in tests.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        ColorFormatter("%(asctime)s [%(levelname)s] %(message)s", use_color=use_color)
    )
    logger.addHandler(console_handler)

    if log_dir:
        path = Path(str(log_dir))
        path.mkdir(parents=True, exist_ok=True)
        file_handler: logging.Handler
        file_handler = RotatingFileHandler(
            path / "llmplotbot.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        if json_logs:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
        logger.addHandler(file_handler)

    return logger


def _coerce_level(level: object) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        value = getattr(logging, level.upper(), None)
        if isinstance(value, int):
            return value
    return logging.INFO


__all__ = ["configure_logging"]
