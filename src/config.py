"""Minimal configuration helpers for LLMPlotBot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "GENERATED_DIR": "./data/generated_data",
    "REQUEST_TIMEOUT": 90,
    "WRITE_STRATEGY": "immediate",
    "WRITE_BATCH_SIZE": 25,
    "WRITE_BATCH_SECONDS": 5.0,
    "WRITE_BATCH_RETRY_LIMIT": 3,
    "FILE_LOCK_TIMEOUT": 10.0,
    "FILE_LOCK_POLL_INTERVAL": 0.1,
    "FILE_LOCK_STALE_SECONDS": 300.0,
}


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load overrides from JSON and merge with :data:`DEFAULT_CONFIG`."""

    config = dict(DEFAULT_CONFIG)
    if path is None:
        path = Path("config") / "config.json"
    else:
        path = Path(path)

    if not path.exists():
        return config

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise ValueError("Configuration file must contain a JSON object.")

    config.update(data)
    return config


__all__ = ["DEFAULT_CONFIG", "load_config"]
