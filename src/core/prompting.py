"""Utilities for parsing and validating model responses."""

from __future__ import annotations

import json
from pprint import pformat
from typing import Any, Iterable


def parse_json_payload(text: str) -> Any | None:
    """Attempt to parse ``text`` as JSON, returning ``None`` on failure."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def is_valid_entry(entry: dict) -> bool:
    """Perform a quick sanity check on a JSON entry returned by the model."""

    required = {"core_event", "themes", "tone"}
    return all(key in entry for key in required)


def format_debug_payload(payload: Any) -> str:
    """Render payload data for debug logging."""

    return pformat(payload, sort_dicts=False)


def iter_first_entry(parsed: Any) -> Iterable[dict]:
    """Yield the first dictionary entry from a parsed payload, if any."""

    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, dict):
            yield first


__all__ = [
    "parse_json_payload",
    "is_valid_entry",
    "format_debug_payload",
    "iter_first_entry",
]
