"""Helpers for loading Reddit headline indices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class Headline:
    identifier: str
    title: str
    source_path: str | None = None


def load_titles(path: str | Path) -> List[Headline]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Titles index not found: {file_path}")
    data = json.loads(file_path.read_text(encoding="utf-8"))
    headlines: List[Headline] = []
    if isinstance(data, dict):
        iterable: Iterable[tuple[str, object]] = data.items()
    elif isinstance(data, list):
        iterable = enumerate(data)
    else:
        raise ValueError("Titles index must be a JSON object or array")
    for key, value in iterable:
        if isinstance(value, dict):
            identifier = str(value.get("id") or key)
            title = str(value.get("title") or value.get("headline") or "")
            source = value.get("file_path")
        else:
            identifier = str(key if isinstance(key, str) else value)
            title = str(value)
            source = None
        if not title:
            continue
        headlines.append(Headline(identifier=identifier, title=title, source_path=source))
    return headlines


__all__ = ["Headline", "load_titles"]
