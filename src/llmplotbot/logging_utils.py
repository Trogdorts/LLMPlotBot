"""Centralised logging configuration helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable


def configure_logging(level: str | int, *, log_dir: str | Path | None = None) -> logging.Logger:
    """Configure the root logger and return the application logger."""

    resolved_level = _coerce_level(level)
    logging.basicConfig(level=resolved_level, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("LLMPlotBot")
    logger.setLevel(resolved_level)

    if log_dir:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path / "llmplotbot.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        file_handler.setLevel(resolved_level)
        logger.addHandler(file_handler)

    return logger


def _coerce_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        try:
            return getattr(logging, level.upper())
        except AttributeError:
            return logging.INFO
    return logging.INFO


__all__ = ["configure_logging"]
