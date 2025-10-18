"""Configuration loading and management utilities for LLMPlotBot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Tuple

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "DATA_DIR": "data",
    "GENERATED_DIR": "data/generated_data",
    "PROMPT_DIR": "data",
    "PROMPT_FILE": "prompt.txt",
    "PROMPT_ARCHIVE_DIR": "data/prompts",
    "TITLES_INDEX": "data/titles_index.json",
    "LOG_DIR": "logs",
    "BACKUP_DIR": "backups",
    "LLM_BASE_URL": "http://localhost:1234",
    "LLM_MODELS": [],
    "LLM_ENDPOINTS": {},
    "LLM_BLOCKLIST": [],
    "REQUEST_TIMEOUT": 90,
    "TASK_BATCH_SIZE": 1,
    "TEST_MODE": False,
    "TEST_LIMIT_PER_MODEL": 10,
    "RETRY_LIMIT": 3,
    "FILE_LOCK_TIMEOUT": 10.0,
    "FILE_LOCK_POLL_INTERVAL": 0.1,
    "FILE_LOCK_STALE_SECONDS": 300.0,
    "COMPLIANCE_REMINDER_INTERVAL": 0,
    "WRITE_STRATEGY": "immediate",
    "WRITE_BATCH_SIZE": 25,
    "WRITE_BATCH_SECONDS": 5.0,
    "WRITE_BATCH_RETRY_LIMIT": 3,
    "LOG_LEVEL": "INFO",
}

CONFIG_DIRNAME = "config"
CONFIG_FILENAME = "config.json"
LOCAL_CONFIG_FILENAME = "config.local.json"
ENV_CONFIG_VARIABLE = "LLMPLOTBOT_CONFIG"


# ---------------------------------------------------------------------------
# Dataclasses describing loaded configuration state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigLoadResult:
    """Final merged configuration along with its contributing sources."""

    config: Dict[str, Any]
    sources: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Attempt to locate the project root by walking upward from this file."""

    current = Path(__file__).resolve()
    for parent in (current,) + tuple(current.parents):
        if (parent / "requirements.txt").exists() and (parent / "main.py").exists():
            return parent
    # fall back to two levels up to keep behaviour stable when packaged
    return Path(__file__).resolve().parents[2]


def _config_search_paths() -> Iterator[Tuple[Path, bool]]:
    """Yield potential configuration files and whether they must exist."""

    root = _project_root()
    config_dir = root / CONFIG_DIRNAME
    default_path = config_dir / CONFIG_FILENAME
    yield default_path, True
    yield config_dir / LOCAL_CONFIG_FILENAME, False
    yield root / LOCAL_CONFIG_FILENAME, False

    env_path = os.getenv(ENV_CONFIG_VARIABLE)
    if env_path:
        yield Path(env_path), False


def _ensure_default_config(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(path)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid user config
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Configuration file {path} must contain a JSON object")
    return dict(data)


def _deep_merge(base: Dict[str, Any], overlay: MutableMapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], MutableMapping)
            and isinstance(value, MutableMapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(*, include_sources: bool = False) -> ConfigLoadResult | Dict[str, Any]:
    """Load configuration defaults merged with user overrides."""

    config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    sources: List[str] = []

    for path, required in _config_search_paths():
        if required:
            _ensure_default_config(path)
        if not path.exists():
            continue
        data = _load_json(path)
        config = _deep_merge(config, data)
        sources.append(str(path.resolve()))

    result = ConfigLoadResult(config=config, sources=tuple(sources))
    if include_sources:
        return result
    return result.config


__all__ = ["DEFAULT_CONFIG", "ConfigLoadResult", "load_config"]
