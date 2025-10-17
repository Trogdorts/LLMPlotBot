"""Input helpers for loading prompts and title metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict


def load_prompt(path: Path, *, logger: logging.Logger | None = None) -> str:
    """Load the prompt template from ``path`` and strip trailing whitespace."""

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if logger:
        logger.debug("Loaded prompt from %s (%s chars).", path, len(content))
    return content


def load_titles(path: Path, *, logger: logging.Logger | None = None) -> Dict[str, Any]:
    """Load the title index JSON structure from ``path``."""

    if not path.exists():
        raise FileNotFoundError(f"Title index not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if logger:
        logger.info("Loaded %s title entries from %s.", len(data), path)
    return data


__all__ = ["load_prompt", "load_titles"]
