"""Prompt construction helpers."""

from __future__ import annotations

import hashlib


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


def make_structured_prompt(title: str) -> str:
    """Compatibility wrapper for :func:`build_structured_prompt`."""

    return build_structured_prompt(title)


__all__ = ["build_structured_prompt", "hash_prompt", "make_structured_prompt"]

