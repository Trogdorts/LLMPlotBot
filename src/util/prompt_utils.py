
"""Prompt utilities for loading, splitting, and archiving prompts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple


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
  "id": "<unique_id_or_leave_blank>",
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
    "id": "",
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
    "id": "",
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


def _ensure_prompt_files(
    prompt_file: str,
    instructions_path: str,
    formatting_path: str,
    logger,
) -> Tuple[str, str]:
    """Ensure the split prompt files exist and return their contents."""

    # Bootstrap from a legacy single prompt file if the split files do not exist yet.
    if os.path.exists(prompt_file) and (
        not os.path.exists(instructions_path) or not os.path.exists(formatting_path)
    ):
        with open(prompt_file, "r", encoding="utf-8") as legacy:
            legacy_text = legacy.read()
        legacy_text = legacy_text.strip()
        lower = legacy_text.lower()
        marker = "### output rules"
        if marker in lower:
            idx = lower.index(marker)
            dynamic = legacy_text[:idx]
            formatting = legacy_text[idx:]
        else:
            dynamic = legacy_text
            formatting = DEFAULT_FORMATTING
        dynamic = dynamic.strip() or DEFAULT_INSTRUCTIONS
        formatting = formatting.strip() or DEFAULT_FORMATTING
        os.makedirs(os.path.dirname(instructions_path), exist_ok=True)
        with open(instructions_path, "w", encoding="utf-8") as f:
            f.write(dynamic.strip() + "\n")
        with open(formatting_path, "w", encoding="utf-8") as f:
            f.write(formatting.strip() + "\n")
        logger.info(
            "Migrated legacy prompt to split files: %s, %s",
            instructions_path,
            formatting_path,
        )

    if not os.path.exists(instructions_path):
        with open(instructions_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_INSTRUCTIONS + "\n")
        logger.warning("Created default prompt instructions: %s", instructions_path)

    if not os.path.exists(formatting_path):
        with open(formatting_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_FORMATTING + "\n")
        logger.warning("Created default prompt formatting rules: %s", formatting_path)

    with open(instructions_path, "r", encoding="utf-8") as instructions_file:
        instructions_text = instructions_file.read()
    with open(formatting_path, "r", encoding="utf-8") as formatting_file:
        formatting_text = formatting_file.read()

    return instructions_text, formatting_text


def load_and_archive_prompt(base_dir: str, logger) -> PromptBundle:
    """
    Ensure a split prompt exists, combine sections, archive the composite prompt, and
    return both the combined text and section references.
    """

    prompt_file = os.path.join(base_dir, "prompt.txt")
    instructions_path = os.path.join(base_dir, "prompt_instructions.txt")
    formatting_path = os.path.join(base_dir, "prompt_formatting.txt")
    prompts_dir = os.path.join(base_dir, "prompts")
    meta_file = os.path.join(base_dir, "prompt_index.json")
    os.makedirs(prompts_dir, exist_ok=True)

    instructions_text, formatting_text = _ensure_prompt_files(
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
        logger.info("Archived new prompt: %s", archive_path)
    else:
        logger.debug("Prompt already archived: %s", archive_path)

    meta = {
        "prompt": prompt_text,
        "hash": prompt_hash,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dynamic_path": instructions_path,
        "formatting_path": formatting_path,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.debug("Saved prompt metadata: %s", meta_file)

    return PromptBundle(
        prompt=prompt_text,
        prompt_hash=prompt_hash,
        dynamic_section=dynamic_section,
        formatting_section=formatting_section,
    )
