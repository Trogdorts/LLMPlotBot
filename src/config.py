"""Configuration loading with layered override support."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

DEFAULT_CONFIG: Dict[str, Any] = {
    "BASE_DIR": "./data",
    "BACKUP_DIR": "./backups",
    "LOG_DIR": "./logs",
    "GENERATED_DIR": "./data/generated_data",
    "IGNORE_FOLDERS": ["backups", ".venv", "__pycache__", "logs"],
    "TEST_MODE": True,
    "TEST_LIMIT_PER_MODEL": 10,
    "JSON_DIR": "./data/json",
    "MAX_WORKERS": 8,
    "RETRY_LIMIT": 3,
    "REQUEST_TIMEOUT": 90,
    "LLM_BASE_URL": "http://localhost:1234",
    "LLM_MODELS": [],
    "LLM_ENDPOINTS": {},
    "LLM_BLOCKLIST": [],
    "WRITE_STRATEGY": "immediate",
    "WRITE_BATCH_SIZE": 25,
    "WRITE_BATCH_SECONDS": 5.0,
    "FILE_LOCK_TIMEOUT": 10.0,
    "FILE_LOCK_POLL_INTERVAL": 0.1,
    "FILE_LOCK_STALE_SECONDS": 300.0,
    "COMPLIANCE_REMINDER_INTERVAL": 0,
    "SUMMARY_LOG_INTERVAL_SECONDS": 0,
}

_OVERRIDE_FILENAMES = ("config.local.json",)
_ENV_OVERRIDE = "LLMPLOTBOT_CONFIG"


def _merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            base[key] = _merge_dicts(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _load_override(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config override at {path} must contain a JSON object.")
    return data


def load_config(
    *,
    defaults: Dict[str, Any] | None = None,
    override_paths: Iterable[str] | None = None,
    include_sources: bool = False,
) -> Tuple[Dict[str, Any], Tuple[str, ...]] | Dict[str, Any]:
    """Return configuration merged with user overrides."""

    config = deepcopy(defaults or DEFAULT_CONFIG)
    sources: list[str] = []

    search_paths = []
    if override_paths:
        search_paths.extend(override_paths)

    project_root = Path(__file__).resolve().parents[1]
    search_paths.extend(str(project_root / name) for name in _OVERRIDE_FILENAMES)

    env_path = os.getenv(_ENV_OVERRIDE)
    if env_path:
        search_paths.append(env_path)

    seen: set[Path] = set()
    for raw_path in search_paths:
        try:
            candidate = Path(raw_path).expanduser()
        except Exception:
            continue
        resolved = candidate.resolve(strict=False)
        if resolved in seen or not candidate.is_file():
            continue
        try:
            overrides = _load_override(candidate)
        except Exception:
            continue
        config = _merge_dicts(config, overrides)
        sources.append(resolved.as_posix())
        seen.add(resolved)

    if include_sources:
        return config, tuple(sources)
    return config


CONFIG, CONFIG_SOURCES = load_config(include_sources=True)


__all__ = ["CONFIG", "CONFIG_SOURCES", "DEFAULT_CONFIG", "load_config"]
