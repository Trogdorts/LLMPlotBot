"""Prompt utilities that assemble instructions from a structured specification."""

from __future__ import annotations

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


PROMPT_FILENAME = "prompt.txt"
PROMPT_ARCHIVE_DIRNAME = "prompts"
PROMPT_META_FILENAME = "prompt_index.json"


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


def _split_prompt_sections(prompt_text: str) -> tuple[str, str]:
    """Split author instructions into dynamic and formatting sections."""

    normalised = _normalise(prompt_text)
    if not normalised:
        return "", ""

    markers = (
        "\n### RESPONSE FORMAT",
        "\n### Output Format",
        "\n### OUTPUT RULES",
        "\nOUTPUT RULES",
    )

    for marker in markers:
        index = normalised.find(marker)
        if index != -1:
            dynamic = normalised[:index].strip()
            formatting = normalised[index + 1 :].strip()
            return dynamic, formatting

    return normalised, ""


def load_and_archive_prompt(base_dir: str, logger) -> PromptBundle:
    """Load prompt instructions from disk (or create defaults) and archive them."""

    spec = DEFAULT_PROMPT_SPECIFICATION
    os.makedirs(base_dir, exist_ok=True)
    prompt_path = os.path.join(base_dir, PROMPT_FILENAME)

    prompt_text = ""
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as handle:
            prompt_text = _normalise(handle.read())
        if prompt_text:
            logger.debug(
                "Loaded prompt instructions from %s",
                normalize_for_logging(prompt_path, extra_roots=[base_dir]),
            )
        else:
            logger.warning(
                "Prompt file at %s was empty; recreating defaults.",
                normalize_for_logging(prompt_path, extra_roots=[base_dir]),
            )

    if not prompt_text:
        builder = PromptBuilder(specification=spec)
        dynamic_default, formatting_default, combined_default = builder.build()
        prompt_text = combined_default
        logger.info(
            "Created default prompt at %s",
            normalize_for_logging(prompt_path, extra_roots=[base_dir]),
        )
        dynamic_section = dynamic_default
        formatting_section = formatting_default
    else:
        dynamic_section, formatting_section = _split_prompt_sections(prompt_text)

    combined = "\n\n".join(
        section for section in (dynamic_section, formatting_section) if section
    )

    prompt_hash = _short_hash(combined)
    _write_prompt_files(base_dir, combined, prompt_hash, spec, logger)

    return PromptBundle(
        prompt=combined,
        prompt_hash=prompt_hash,
        dynamic_section=dynamic_section,
        formatting_section=formatting_section,
        specification=spec,
    )


__all__ = [
    "PromptBundle",
    "load_and_archive_prompt",
]
