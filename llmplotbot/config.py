"""Configuration loading and validation for LLMPlotBot."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
ENV_CONFIG_PATH = "LLMPLOTBOT_CONFIG"


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "timeout": 120,
        "max_concurrency": 4,
        "retry": {
            "max_attempts": 4,
            "backoff_seconds": 2.0,
            "max_backoff_seconds": 30.0,
        },
        "models": ["creative-writing"],
    },
    "paths": {
        "data": "data",
        "outputs": "data/outputs",
        "failed": "data/failed",
        "checkpoints": "data/checkpoints",
        "metrics": "data/metrics",
        "summaries": "data/summaries",
        "logs": "logs",
        "prompts": "prompts",
        "prompt_archive": "prompts/archive",
        "jobs_db": "data/jobs.db",
        "titles_index": "data/titles_index.json",
        "titles_source": r"C:\\Users\\criss\\OneDrive\\Desktop\\NTO_data\\json",
    },
    "logging": {
        "console_level": "INFO",
        "file_level": "DEBUG",
        "json_logs": False,
        "color": True,
    },
    "metrics": {
        "report_interval": 10,
        "include_system": True,
    },
    "testing": {
        "enabled": False,
        "dry_run": False,
        "max_jobs": 50,
    },
    "checkpoints": {
        "interval_seconds": 30,
        "jobs_per_checkpoint": 25,
    },
    "system": {
        "integrity_check_interval": 300,
    },
}


@dataclass(frozen=True)
class ConfigLoadResult:
    """Container for the merged configuration."""

    config: Dict[str, Any]
    sources: Tuple[str, ...]


def _ensure_default_config(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(DEFAULT_CONFIG, handle, sort_keys=True)
    tmp.replace(path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Configuration file {path} must contain a mapping at the root")
    return dict(data)


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, MutableMapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Treat Windows drive-letter and UNC paths as already absolute so they
    # are not resolved relative to the project directory on Unix platforms.
    if len(value) >= 2 and value[1] == ":":
        return Path(value)
    if value.startswith("\\\\") or value.startswith("//"):
        return Path(value)
    return (base_dir / value).resolve()


def _apply_path_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = dict(config.get("paths", {}))
    base_dir = PROJECT_ROOT
    for key, rel_path in paths.items():
        if isinstance(rel_path, str):
            paths[key] = str(_resolve_path(base_dir, rel_path))
    config["paths"] = paths
    return config


def _ensure_directories(config: Mapping[str, Any]) -> None:
    path_config = config.get("paths", {})
    for key in ("data", "outputs", "failed", "checkpoints", "metrics", "summaries", "logs", "prompts"):
        value = path_config.get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)
    archive = path_config.get("prompt_archive")
    if archive:
        Path(archive).mkdir(parents=True, exist_ok=True)
    jobs_db = path_config.get("jobs_db")
    if jobs_db:
        Path(jobs_db).parent.mkdir(parents=True, exist_ok=True)


def _collect_sources() -> Iterable[Tuple[Path, bool]]:
    yield CONFIG_PATH, True
    env_path = os.getenv(ENV_CONFIG_PATH)
    if env_path:
        yield Path(env_path), False


def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if "model" not in config:
        raise ValueError("Configuration must define a 'model' section")
    models = config["model"].get("models")
    if not models:
        raise ValueError("At least one model must be configured in model.models")
    timeout = config["model"].get("timeout")
    if timeout is None or int(timeout) <= 0:
        raise ValueError("model.timeout must be a positive integer")
    checkpoints = config.get("checkpoints", {})
    if int(checkpoints.get("interval_seconds", 0)) <= 0:
        raise ValueError("checkpoints.interval_seconds must be > 0")
    return config


def load_config(*, include_sources: bool = False) -> ConfigLoadResult | Dict[str, Any]:
    """Load and validate configuration settings."""

    config: Dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))
    sources: list[str] = []

    for path, required in _collect_sources():
        if required:
            _ensure_default_config(path)
        if not path.exists():
            continue
        data = _load_yaml(path)
        config = _deep_merge(config, data)
        sources.append(str(path.resolve()))

    config = json.loads(json.dumps(config))  # deep copy via JSON for immutability
    config = _apply_path_defaults(config)
    _ensure_directories(config)
    config = _validate_config(config)

    result = ConfigLoadResult(config=config, sources=tuple(sources))
    if include_sources:
        return result
    return result.config


__all__ = ["DEFAULT_CONFIG", "ConfigLoadResult", "load_config"]
