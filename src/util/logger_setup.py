"""
Logging utilities: colorized console and rotating file logs.
Always preserves a complete DEBUG log in logs/debug.log.
"""

import logging
from typing import Any


def resolve_log_level(level: Any, *, default: int = logging.INFO) -> int:
    """Return a valid logging level for ``level``.

    Accepts standard logging level names (case insensitive), integers, or
    ``None``. Any invalid input falls back to ``default``. This helper keeps
    logging configuration resilient when values come from configuration files.
    """

    if isinstance(level, int):
        return level
    if isinstance(level, str):
        candidate = getattr(logging, level.upper(), None)
        if isinstance(candidate, int):
            return candidate
    return default


__all__ = ["resolve_log_level"]
