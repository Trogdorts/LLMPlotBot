
"""
Prompt utilities: manage active prompt, hash, and archival index.
"""

import os
import hashlib
import json
from datetime import datetime


def _short_hash(text: str, length: int = 8) -> str:
    """Return a short SHA256 prefix for compact indexing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def load_and_archive_prompt(base_dir: str, logger):
    """
    Ensure data/prompt.txt exists, read it, compute short hash, archive under data/prompts/{hash}.txt,
    and save metadata to data/prompt_index.json.
    """
    prompt_file = os.path.join(base_dir, "prompt.txt")
    prompts_dir = os.path.join(base_dir, "prompts")
    meta_file = os.path.join(base_dir, "prompt_index.json")
    os.makedirs(prompts_dir, exist_ok=True)

    if not os.path.exists(prompt_file):
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write("Enter your prompt here.")
        logger.warning(f"Created default prompt file: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    prompt_hash = _short_hash(prompt_text)

    archive_path = os.path.join(prompts_dir, f"{prompt_hash}.txt")
    if not os.path.exists(archive_path):
        with open(archive_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        logger.info(f"Archived new prompt: {archive_path}")
    else:
        logger.debug("Prompt already archived.")

    meta = {"prompt": prompt_text, "hash": prompt_hash, "timestamp": datetime.now().isoformat(timespec="seconds")}
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.debug(f"Saved prompt metadata: {meta_file}")

    return {"prompt": prompt_text, "hash": prompt_hash}
