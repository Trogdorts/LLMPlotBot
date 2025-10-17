"""Prompt utilities that assemble instructions from a structured specification."""

from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime

from src.core.prompt_spec import (
    DEFAULT_PROMPT_SPECIFICATION,
    PromptBuilder,
    PromptSpecification,
)
from .path_utils import normalize_for_logging


USER_PROMPT_FILENAME = "prompt_user_snippet.txt"
PROMPT_FILENAME = "prompt.txt"
PROMPT_ARCHIVE_DIRNAME = "prompts"
PROMPT_META_FILENAME = "prompt_index.json"


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
def make_structured_prompt(title: str) -> str:
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


def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def validate_entry(entry: dict) -> bool:
    """Quick sanity check on a single parsed JSON entry."""
    required = {"core_event", "themes", "tone"}
    return all(k in entry for k in required)

@dataclass
class PromptBundle:
    """Structured prompt payload returned to the processing pipeline."""

    prompt: str
    prompt_hash: str
    dynamic_section: str
    formatting_section: str
    specification: PromptSpecification


def _short_hash(text: str, length: int = 8) -> str:
    """Return a short SHA256 prefix for compact indexing."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _normalise(text: str) -> str:
    """Normalise newline characters and trim whitespace."""

    if not isinstance(text, str):
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _ensure_user_prompt(base_dir: str, logger, spec: PromptSpecification) -> str:
    """Ensure a user-adjustable prompt snippet exists and return its contents."""

    os.makedirs(base_dir, exist_ok=True)
    user_prompt_path = os.path.join(base_dir, USER_PROMPT_FILENAME)

    if os.path.exists(user_prompt_path):
        with open(user_prompt_path, "r", encoding="utf-8") as handle:
            snippet = _normalise(handle.read())
        if snippet:
            logger.debug(
                "Loaded user prompt snippet from %s",
                normalize_for_logging(user_prompt_path, extra_roots=[base_dir]),
            )
            return snippet
        logger.warning(
            "User prompt snippet at %s was empty; restoring default instructions.",
            normalize_for_logging(user_prompt_path, extra_roots=[base_dir]),
        )

    default_text = spec.default_user_prompt.strip()
    with open(user_prompt_path, "w", encoding="utf-8") as handle:
        handle.write(default_text + "\n")
    logger.info(
        "Created default user prompt snippet at %s",
        normalize_for_logging(user_prompt_path, extra_roots=[base_dir]),
    )
    return default_text


def _write_prompt_files(
    base_dir: str,
    prompt_text: str,
    prompt_hash: str,
    spec: PromptSpecification,
    logger,
) -> None:
    """Persist the generated prompt to disk along with archival metadata."""

    prompt_file = os.path.join(base_dir, PROMPT_FILENAME)
    prompts_dir = os.path.join(base_dir, PROMPT_ARCHIVE_DIRNAME)
    meta_file = os.path.join(base_dir, PROMPT_META_FILENAME)

    os.makedirs(prompts_dir, exist_ok=True)

    with open(prompt_file, "w", encoding="utf-8") as handle:
        handle.write(prompt_text + "\n")

    archive_path = os.path.join(prompts_dir, f"{prompt_hash}.txt")
    if not os.path.exists(archive_path):
        with open(archive_path, "w", encoding="utf-8") as archive_handle:
            archive_handle.write(prompt_text)
        logger.info(
            "Archived prompt %s",
            normalize_for_logging(archive_path, extra_roots=[base_dir]),
        )
    else:
        logger.debug(
            "Prompt %s already archived.",
            normalize_for_logging(archive_path, extra_roots=[base_dir]),
        )

    field_meta = [
        {
            "name": field.name,
            "type": "array" if field.is_list() else "string",
            "description": field.description,
            "constraints": list(field.constraints),
        }
        for field in spec.fields
    ]

    meta_payload = {
        "prompt_hash": prompt_hash,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_path": prompt_file,
        "user_prompt_path": os.path.join(base_dir, USER_PROMPT_FILENAME),
        "fields": field_meta,
        "formatting_rules": list(spec.formatting_rules),
        "quality_checks": list(spec.quality_checks),
    }
    with open(meta_file, "w", encoding="utf-8") as meta_handle:
        json.dump(meta_payload, meta_handle, ensure_ascii=False, indent=2)
        meta_handle.write("\n")
    logger.debug(
        "Saved prompt metadata to %s",
        normalize_for_logging(meta_file, extra_roots=[base_dir]),
    )


def load_and_archive_prompt(base_dir: str, logger) -> PromptBundle:
    """Generate prompt sections from the default specification and persist them."""

    spec = DEFAULT_PROMPT_SPECIFICATION
    user_snippet = _ensure_user_prompt(base_dir, logger, spec)
    builder = PromptBuilder(specification=spec, user_prompt=user_snippet)
    dynamic, formatting, combined = builder.build()

    prompt_hash = _short_hash(combined)
    _write_prompt_files(base_dir, combined, prompt_hash, spec, logger)

    return PromptBundle(
        prompt=combined,
        prompt_hash=prompt_hash,
        dynamic_section=dynamic,
        formatting_section=formatting,
        specification=spec,
    )


__all__ = [
    "PromptBundle",
    "load_and_archive_prompt",
]
