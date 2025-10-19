"""Prompt loading and hashing utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class PromptBundle:
    prompt: str
    prompt_hash: str
    source_path: Path


class PromptManager:
    """Loads prompt text and ensures deterministic hashing."""

    def __init__(self, prompt_dir: str, *, filename: str, archive_dir: str | None = None) -> None:
        self.prompt_dir = Path(prompt_dir)
        self.filename = filename
        self.archive_dir = Path(archive_dir) if archive_dir else None

    def load(self) -> PromptBundle:
        path = self.prompt_dir / self.filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        prompt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        bundle = PromptBundle(prompt=text, prompt_hash=prompt_hash, source_path=path)
        self._archive_prompt(bundle)
        return bundle

    def _archive_prompt(self, bundle: PromptBundle) -> None:
        if not self.archive_dir:
            return
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        archive_path = self.archive_dir / f"prompt-{timestamp}-{bundle.prompt_hash[:12]}.txt"
        archive_path.write_text(bundle.prompt + "\n", encoding="utf-8")


__all__ = ["PromptBundle", "PromptManager"]
