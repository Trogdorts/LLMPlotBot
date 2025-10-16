
"""Prompt utilities for loading, splitting, and archiving prompts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

from .path_utils import normalize_for_logging


DEFAULT_INSTRUCTIONS = """You are a story-idea abstraction engine.

Your job is to convert multiple real-world news headlines into structured story-seed records.
You must always return a single valid JSON array `[...]`.
The array must parse cleanly with no trailing commas or text outside the brackets.

---

### INPUT
You will receive several numbered headlines, one per line:
1. Headline text A
2. Headline text B
3. Headline text C
(etc.)

---

### TASK
For **each headline**, independently:
1. Rewrite it as one complete, natural-sounding sentence under 50 words.
   - Keep the original irony, tone, and absurdity.
   - Replace names with neutral roles or archetypes only if the name is not essential.
   - Replace organizations or places with neutral equivalents if possible.
   - Remove dates unless vital.
   - **Do not** invent or remove facts.

2. Fill in these abstract elements:
   - `themes` — 2-5 short conceptual keywords.
   - `tone` — one short stylistic label.
   - `conflict_type` — 1-3 words for the central tension.
   - `stakes` — one sentence describing what is at risk or changing.
   - `setting_hint` — a brief situational cue (e.g., “rural village,” “tech startup”).
   - `characters` — 2-5 archetypal roles.
   - `potential_story_hooks` — 1-3 concise ideas for how the story could continue.

3. Output **one JSON object** per headline using exactly this schema:

{
  "core_event": "<rewritten_sentence>",
  "themes": ["theme1","theme2"],
  "tone": "<tone>",
  "conflict_type": "<conflict_type>",
  "stakes": "<stakes>",
  "setting_hint": "<setting_hint>",
  "characters": ["role1","role2"],
  "potential_story_hooks": ["hook1","hook2"]
}
"""


DEFAULT_FORMATTING = """---

### OUTPUT RULES
- Return **only** a single JSON array containing one object per headline, in order.
- Do **not** wrap the array in quotes or markdown.
- Always include these keys: `core_event`, `themes`, `tone`, `conflict_type`, `stakes`,
  `setting_hint`, `characters`, and `potential_story_hooks`.
- Include **all keys** even if some arrays are empty (`[]`) or values are empty strings (`""`).
- No commentary, explanation, or prose before or after the JSON.
- Double-check that brackets and commas make valid JSON.
- Before returning, recheck that your JSON is strictly valid.
  Remove stray commas, mismatched quotes, or dangling punctuation that
  would cause a JSON parser to fail. Output only the corrected JSON array.

---

### Example Input
1. Scientists accidentally create AI that refuses to stop making motivational quotes.
2. Mayor bans meetings after complaints about too many meetings.

### Example Output
[
  {
    "core_event": "Researchers accidentally develop an AI that endlessly generates motivational quotes and refuses shutdown.",
    "themes": ["technology","hubris","identity"],
    "tone": "satirical",
    "conflict_type": "creation vs control",
    "stakes": "an uncontrollable AI floods the world with unwanted optimism",
    "setting_hint": "tech startup lab",
    "characters": ["researcher","AI system","executive","technician"],
    "potential_story_hooks": ["AI begins inspiring cult-like followers","researchers debate deleting their creation"]
  },
  {
    "core_event": "A city mayor cancels all meetings after complaints about excessive meetings, leaving the administration unable to function.",
    "themes": ["bureaucracy","absurdity"],
    "tone": "ironic",
    "conflict_type": "policy backlash",
    "stakes": "government operations grind to a halt",
    "setting_hint": "city hall",
    "characters": ["mayor","staff","citizens"],
    "potential_story_hooks": ["a city trapped in silence","bureaucrats rebel against enforced idleness"]
  }
]
"""


@dataclass
class PromptBundle:
    """Structured prompt payload with separated dynamic/static sections."""

    prompt: str
    prompt_hash: str
    dynamic_section: str
    formatting_section: str


def _short_hash(text: str, length: int = 8) -> str:
    """Return a short SHA256 prefix for compact indexing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _normalize_section(text: str) -> str:
    """Normalize newline characters and trim surrounding whitespace."""
    if not isinstance(text, str):
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _split_prompt_sections(text: str) -> Tuple[str, str]:
    """Split a combined prompt into instructions and formatting sections."""

    normalized = _normalize_section(text)
    if not normalized:
        return DEFAULT_INSTRUCTIONS.strip(), DEFAULT_FORMATTING.strip()

    lower = normalized.lower()
    marker = "### output rules"
    if marker in lower:
        idx = lower.index(marker)
        instructions = normalized[:idx].rstrip()
        formatting = normalized[idx:].lstrip()
    else:
        instructions = normalized
        formatting = DEFAULT_FORMATTING.strip()

    instructions = instructions or DEFAULT_INSTRUCTIONS.strip()
    formatting = formatting or DEFAULT_FORMATTING.strip()
    return instructions, formatting


def _ensure_prompt_file(
    prompt_file: str,
    instructions_path: str,
    formatting_path: str,
    logger,
) -> Tuple[str, str]:
    """Ensure a single prompt file exists and return its split sections."""

    instructions_text = ""
    formatting_text = ""
    base_dir = os.path.dirname(os.path.abspath(prompt_file))

    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
        instructions_text, formatting_text = _split_prompt_sections(prompt_text)
    elif os.path.exists(instructions_path) or os.path.exists(formatting_path):
        # Migrate any previously split files into the unified prompt.txt.
        if os.path.exists(instructions_path):
            with open(instructions_path, "r", encoding="utf-8") as f:
                instructions_text = _normalize_section(f.read())
        if os.path.exists(formatting_path):
            with open(formatting_path, "r", encoding="utf-8") as f:
                formatting_text = _normalize_section(f.read())
        instructions_text = instructions_text or DEFAULT_INSTRUCTIONS.strip()
        formatting_text = formatting_text or DEFAULT_FORMATTING.strip()
        logger.info(
            "Migrated split prompt files into %s",
            normalize_for_logging(prompt_file, extra_roots=[base_dir]),
        )
    else:
        instructions_text = DEFAULT_INSTRUCTIONS.strip()
        formatting_text = DEFAULT_FORMATTING.strip()
        logger.warning(
            "Created default prompt content in %s",
            normalize_for_logging(prompt_file, extra_roots=[base_dir]),
        )

    combined_prompt = "\n\n".join(
        section.strip() for section in (instructions_text, formatting_text) if section
    ).strip()

    os.makedirs(os.path.dirname(prompt_file), exist_ok=True)
    with open(prompt_file, "w", encoding="utf-8") as combined_file:
        combined_file.write(combined_prompt + "\n")

    # Remove any legacy split files so they are not recreated on subsequent runs.
    for legacy_path in (instructions_path, formatting_path):
        if os.path.exists(legacy_path):
            try:
                os.remove(legacy_path)
                logger.info(
                    "Removed legacy prompt file: %s",
                    normalize_for_logging(legacy_path, extra_roots=[base_dir]),
                )
            except OSError:
                logger.debug(
                    "Unable to remove legacy prompt file: %s",
                    normalize_for_logging(legacy_path, extra_roots=[base_dir]),
                )

    return instructions_text, formatting_text


def load_and_archive_prompt(base_dir: str, logger) -> PromptBundle:
    """Return the active prompt bundle while keeping on-disk artefacts in sync."""

    prompt_file = os.path.join(base_dir, "prompt.txt")
    instructions_path = os.path.join(base_dir, "prompt_instructions.txt")
    formatting_path = os.path.join(base_dir, "prompt_formatting.txt")
    prompts_dir = os.path.join(base_dir, "prompts")
    meta_file = os.path.join(base_dir, "prompt_index.json")
    os.makedirs(prompts_dir, exist_ok=True)

    instructions_text, formatting_text = _ensure_prompt_file(
        prompt_file, instructions_path, formatting_path, logger
    )

    dynamic_section = _normalize_section(instructions_text)
    formatting_section = _normalize_section(formatting_text)

    prompt_text = "\n\n".join(
        section for section in (dynamic_section, formatting_section) if section
    ).strip()

    # Keep prompt.txt synchronised for backwards compatibility / manual review.
    with open(prompt_file, "w", encoding="utf-8") as combined_file:
        combined_file.write(prompt_text + "\n")

    prompt_hash = _short_hash(prompt_text)

    archive_path = os.path.join(prompts_dir, f"{prompt_hash}.txt")
    if not os.path.exists(archive_path):
        with open(archive_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        logger.info(
            "Archived new prompt: %s",
            normalize_for_logging(archive_path, extra_roots=[base_dir]),
        )
    else:
        logger.debug(
            "Prompt already archived: %s",
            normalize_for_logging(archive_path, extra_roots=[base_dir]),
        )

    meta = {
        "prompt": prompt_text,
        "hash": prompt_hash,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt_path": prompt_file,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.debug(
        "Saved prompt metadata: %s",
        normalize_for_logging(meta_file, extra_roots=[base_dir]),
    )

    return PromptBundle(
        prompt=prompt_text,
        prompt_hash=prompt_hash,
        dynamic_section=dynamic_section,
        formatting_section=formatting_section,
    )
