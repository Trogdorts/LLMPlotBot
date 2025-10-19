"""Helpers for loading and generating Reddit headline indices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple


@dataclass(frozen=True)
class Headline:
    identifier: str
    title: str
    source_path: str | None = None


def load_titles(path: str | Path, *, source_dir: str | Path | None = None) -> List[Headline]:
    file_path = Path(path)
    if not file_path.exists():
        if not source_dir:
            raise FileNotFoundError(f"Titles index not found: {file_path}")
        _regenerate_titles_index(file_path, Path(source_dir))
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


def _regenerate_titles_index(index_path: Path, source_dir: Path) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(
            f"Titles index missing and source directory not found: {source_dir}"
        )

    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for json_file in sorted(source_dir.glob("*.json")):
        raw = json.loads(json_file.read_text(encoding="utf-8"))
        for counter, (identifier, title) in enumerate(_extract_entries(raw)):
            identifier = identifier.strip()
            title = title.strip()
            if not title:
                continue
            if not identifier:
                identifier = f"{json_file.stem}-{counter}"
            if identifier in seen:
                continue
            seen.add(identifier)
            entries.append(
                {
                    "id": identifier,
                    "title": title,
                    "file_path": str(json_file),
                }
            )

    if not entries:
        raise ValueError(
            f"Unable to regenerate titles index; no valid entries found in {source_dir}"
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _extract_entries(data: object) -> Iterator[Tuple[str, str]]:
    if isinstance(data, dict):
        identifier = data.get("id")
        title = data.get("title") or data.get("headline")
        if isinstance(title, str):
            yield str(identifier or ""), title
        for value in data.values():
            yield from _extract_entries(value)
    elif isinstance(data, list):
        for value in data:
            yield from _extract_entries(value)


__all__ = ["Headline", "load_titles"]
