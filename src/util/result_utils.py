"""Helpers for inspecting existing generated result files."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


class ExistingResultChecker:
    """Inspect generated result files to detect already completed tasks."""

    def __init__(self, base_dir: str, logger=None) -> None:
        self.base = Path(base_dir)
        self.logger = logger

    @lru_cache(maxsize=None)
    def _load_record(self, identifier: str) -> Dict[str, Any]:
        """Load and cache the JSON payload for ``identifier`` if it exists."""

        path = self.base / f"{identifier}.json"
        if not path.exists():
            return {}

        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle) or {}
        except Exception as exc:  # pragma: no cover - defensive logging
            if self.logger:
                self.logger.warning(
                    "Unable to read existing results for %s: %s", identifier, exc
                )
            return {}

    def record_exists(self, identifier: str) -> bool:
        """Return ``True`` if a generated JSON record already exists on disk."""

        path = self.base / f"{identifier}.json"
        return path.exists()

    def has_model_entry(self, identifier: str, model: str) -> bool:
        """Return ``True`` if ``identifier`` includes any payload for ``model``."""

        record = self._load_record(str(identifier))
        models: Optional[Dict[str, Any]] = record.get("llm_models")
        if not isinstance(models, dict):
            return False

        data = models.get(model)
        if isinstance(data, dict):
            return bool(data)
        if isinstance(data, (list, set, tuple)):
            return bool(data)
        return data not in (None, "", False, 0)

    def has_entry(self, identifier: str, model: str, prompt_hash: str) -> bool:
        """Return ``True`` if a result already exists for the given key triple."""

        record = self._load_record(str(identifier))
        models: Optional[Dict[str, Any]] = record.get("llm_models")
        if not isinstance(models, dict):
            return False

        model_results: Optional[Dict[str, Any]] = models.get(model)
        if not isinstance(model_results, dict):
            return False

        return prompt_hash in model_results

