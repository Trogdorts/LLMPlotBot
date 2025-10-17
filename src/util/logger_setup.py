"""
Logging utilities: colorized console and rotating file logs.
Always preserves a complete DEBUG log in logs/debug.log.
"""

import copy
import logging
import os
from logging.handlers import RotatingFileHandler
from pprint import pformat

C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BLUE = "\033[94m"
C_RESET = "\033[0m"


def resolve_log_level(level: Any, *, default: int = logging.INFO) -> int:
    """Return a numeric logging level from a user-provided value."""

    if level is None:
        return default

    if isinstance(level, int):
        return level

    if isinstance(level, str):
        normalized = level.strip()
        if not normalized:
            return default

        if normalized.isdigit():
            return int(normalized)

        upper = normalized.upper()
        level_value = getattr(logging, upper, None)
        if isinstance(level_value, int):
            return level_value

    raise ValueError(f"Unsupported log level value: {level!r}")


def setup_logger(
    log_dir: str,
    console_level: int = logging.DEBUG,
    *,
    file_level: int | None = None,
) -> logging.Logger:
    """
    Create or return the global logger with color console and rotating files.
    The debug file always captures full DEBUG output regardless of console level.

    Args:
        log_dir: Directory where log files should be written.
        console_level: Logging level threshold for console output.
        file_level: Optional logging level threshold for ``app.log``; defaults
            to ``console_level`` when ``None``.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")
    debug_path = os.path.join(log_dir, "debug.log")

    def _maybe_pretty(value):
        if isinstance(value, (dict, list, tuple, set)):
            return pformat(value, width=120, compact=True, sort_dicts=True)
        return value

    def _transform_args(args):
        if isinstance(args, tuple):
            return tuple(_maybe_pretty(arg) for arg in args)
        if isinstance(args, dict):
            return {key: _maybe_pretty(value) for key, value in args.items()}
        return _maybe_pretty(args)

    class ColorFormatter(logging.Formatter):
        def format(self, record):
            # Only mutate a copy so the record stays untouched for other handlers.
            record = copy.copy(record)
            if record.levelno <= logging.DEBUG:
                record.msg = _maybe_pretty(record.msg)
                if record.args:
                    record.args = _transform_args(record.args)
            color = {logging.INFO: C_BLUE, logging.WARNING: C_YELLOW, logging.ERROR: C_RED}.get(record.levelno, "")
            msg = super().format(record)
            return f"{color}{msg}{C_RESET}"

    resolved_file_level = file_level if file_level is not None else console_level

    logger = logging.getLogger("llm_batch_processor")
    if logger.handlers:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_level)
            elif isinstance(handler, RotatingFileHandler):
                base_name = os.path.basename(getattr(handler, "baseFilename", ""))
                if base_name == "debug.log":
                    handler.setLevel(logging.DEBUG)
                else:
                    handler.setLevel(resolved_file_level)
        return logger

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(resolved_file_level)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    dh = RotatingFileHandler(debug_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    dh.setLevel(logging.DEBUG)
    dh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(dh)

    logger.debug("Logger initialized at DEBUG level.")
    return logger
