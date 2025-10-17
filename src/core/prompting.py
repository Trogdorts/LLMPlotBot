"""Utilities for constructing prompts and parsing model responses."""

from __future__ import annotations

import hashlib
import json
from pprint import pformat
from typing import Any, Iterable


def build_structured_prompt(title: str) -> str:
    """Return the structured prompt sent to the language model."""

    return f"""
You are a story-idea abstraction engine.
Fill in the following JSON structure completely based on the given title.
Write natural, complete, realistic content for every field.
Return ONLY valid JSON. Do not add commentary, markdown, or explanation.

Title:
"{title}"

Required output schema:
[
  {{
    "title": "{title}",
    "core_event": "<one complete rewritten sentence under 50 words>",
    "themes": ["concept1", "concept2"],
    "tone": "<stylistic tone label>",
    "conflict_type": "<short phrase for the central tension>",
    "stakes": "<one concise sentence of what’s at risk>",
    "setting_hint": "<short location or situational hint>",
    "characters": ["role1", "role2"],
    "potential_story_hooks": ["hook1", "hook2"]
  }}
]
Output must start with [ and end with ] and be valid JSON.
"""


def hash_prompt(prompt: str) -> str:
    """Generate a stable hash for the given ``prompt``."""

    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def parse_json_payload(text: str) -> Any | None:
    """Attempt to parse ``text`` as JSON, returning ``None`` on failure."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def try_parse_json(text: str) -> Any | None:
    """Compatibility wrapper for :func:`parse_json_payload`."""

    return parse_json_payload(text)


def is_valid_entry(entry: dict) -> bool:
    """Perform a quick sanity check on a JSON entry returned by the model."""

    required = {"core_event", "themes", "tone"}
    return all(key in entry for key in required)


def validate_entry(entry: dict) -> bool:
    """Compatibility wrapper for :func:`is_valid_entry`."""

    return is_valid_entry(entry)


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
    "build_structured_prompt",
    "hash_prompt",
    "parse_json_payload",
    "try_parse_json",
    "is_valid_entry",
    "validate_entry",
    "format_debug_payload",
    "iter_first_entry",
]
