"""Simplified configuration manager — single-file configuration only."""

from __future__ import annotations
import json
import os
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------
# Default configuration (fallback values if config/config.json missing)
# ---------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "BASE_DIR": "./data",
    "BACKUP_DIR": "./backups",
    "LOG_DIR": "./logs",
    "GENERATED_DIR": "./data/generated_data",
    "IGNORE_FOLDERS": ["backups", ".venv", "__pycache__", "logs"],
    "TEST_MODE": False,
    "TEST_LIMIT_PER_MODEL": 10,
    "JSON_DIR": "./data/json",
    "MAX_WORKERS": 8,
    "RETRY_LIMIT": 3,
    "REQUEST_TIMEOUT": 90,
    "LLM_BASE_URL": "http://localhost:1234",
    "LLM_MODELS": [],
    "LLM_ENDPOINTS": {},
    "WRITE_STRATEGY": "immediate",
    "WRITE_BATCH_SIZE": 25,
    "WRITE_BATCH_SECONDS": 5.0,
    "FILE_LOCK_TIMEOUT": 10.0,
    "FILE_LOCK_POLL_INTERVAL": 0.1,
    "FILE_LOCK_STALE_SECONDS": 300.0,
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _resolve_project_root() -> Path:
    """Return the project root (directory containing this file's parent 'src')."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").exists() or (parent / "main.py").exists():
            return parent
    return Path.cwd()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------
def load_config(*, include_sources: bool = False) -> Dict[str, Any] | Tuple[Dict[str, Any], Tuple[str]]:
    """
    Load configuration from a single file at <project_root>/config/config.json.
    If the file does not exist, it will be created with DEFAULT_CONFIG.
    No additional merging or overrides occur.
    """
    project_root = _resolve_project_root()
    config_dir = project_root / "config"
    config_path = config_dir / "config.json"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Load or create config
    if config_path.exists():
        try:
            user_config = _load_json(config_path)
        except Exception as exc:
            raise ValueError(f"Failed to load {config_path}: {exc}") from exc
        merged = deepcopy(DEFAULT_CONFIG)
        merged.update(user_config or {})
    else:
        merged = deepcopy(DEFAULT_CONFIG)
        _write_json(config_path, merged)

    if include_sources:
        return merged, (str(config_path.resolve()),)
    return merged


# ---------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------
CONFIG, CONFIG_SOURCES = load_config(include_sources=True)

__all__ = ["CONFIG", "CONFIG_SOURCES", "DEFAULT_CONFIG", "load_config"]
