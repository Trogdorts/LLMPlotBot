"""Configuration management with automatic default creation."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple

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
    "TASK_BATCH_SIZE": 5,
    "RETRY_LIMIT": 3,
    "REQUEST_TIMEOUT": 90,
    "LLM_BASE_URL": "http://localhost:1234",
    "LLM_MODELS": [],
    "LLM_ENDPOINTS": {},
    "LLM_BLOCKLIST": [],
    "WRITE_RETRY_LIMIT": 3,
    "FILE_LOCK_TIMEOUT": 10.0,
    "FILE_LOCK_POLL_INTERVAL": 0.1,
    "FILE_LOCK_STALE_SECONDS": 300.0,
    "COMPLIANCE_REMINDER_INTERVAL": 0,
    "METRICS_SUMMARY_TASK_INTERVAL": 25,
    "METRICS_SUMMARY_TIME_SECONDS": 120.0,
    "EXPECTED_LANGUAGE": "en",
}

_CONFIG_DIRNAME = "config"
_DEFAULT_CONFIG_FILENAME = "default.json"
_OVERRIDE_FILENAMES = ("config.local.json",)
_ENV_OVERRIDE = "LLMPLOTBOT_CONFIG"


def _merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            base[key] = _merge_dicts(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config data at {path} must be a JSON object.")
    return data


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ConfigManager:
    """Load layered configuration with automatic defaults."""

    defaults: Dict[str, Any]
    project_root: Path
    config_dirname: str = _CONFIG_DIRNAME
    default_filename: str = _DEFAULT_CONFIG_FILENAME
    override_filenames: Tuple[str, ...] = _OVERRIDE_FILENAMES
    env_override: str = _ENV_OVERRIDE

    @property
    def config_dir(self) -> Path:
        return self.project_root / self.config_dirname

    @property
    def default_path(self) -> Path:
        return self.config_dir / self.default_filename

    def ensure_default_config(self) -> Dict[str, Any]:
        """Make sure the default configuration file exists and is up-to-date."""

        self.config_dir.mkdir(parents=True, exist_ok=True)
        base_defaults = deepcopy(self.defaults)

        if self.default_path.exists():
            try:
                stored = _load_json(self.default_path)
            except Exception as exc:  # pragma: no cover - defensive error enrichment
                raise ValueError(
                    f"Unable to load default configuration from {self.default_path}: {exc}"
                ) from exc
            merged = _merge_dicts(base_defaults, stored)
            if merged != stored:
                _write_json(self.default_path, merged)
            return merged

        _write_json(self.default_path, base_defaults)
        return base_defaults

    def _candidate_paths(self, override_paths: Iterable[str] | None) -> Iterator[Path]:
        if override_paths:
            for raw_path in override_paths:
                yield Path(raw_path).expanduser()

        for name in self.override_filenames:
            yield self.config_dir / name

        for name in self.override_filenames:
            yield self.project_root / name

        env_path = os.getenv(self.env_override)
        if env_path:
            yield Path(env_path).expanduser()

    def load(
        self,
        *,
        override_paths: Iterable[str] | None = None,
        include_sources: bool = False,
    ) -> Tuple[Dict[str, Any], Tuple[str, ...]] | Dict[str, Any]:
        config = deepcopy(self.defaults)
        default_values = self.ensure_default_config()

        sources: list[str] = []
        default_resolved = self.default_path.resolve(strict=False)
        config = _merge_dicts(config, default_values)
        sources.append(default_resolved.as_posix())

        seen: set[Path] = {default_resolved}
        for candidate in self._candidate_paths(override_paths):
            try:
                resolved = candidate.resolve(strict=False)
            except Exception:
                continue
            if resolved in seen or not candidate.is_file():
                continue
            try:
                overrides = _load_json(candidate)
            except Exception:
                continue
            config = _merge_dicts(config, overrides)
            sources.append(resolved.as_posix())
            seen.add(resolved)

        if include_sources:
            return config, tuple(sources)
        return config


def load_config(
    *,
    defaults: Dict[str, Any] | None = None,
    override_paths: Iterable[str] | None = None,
    include_sources: bool = False,
) -> Tuple[Dict[str, Any], Tuple[str, ...]] | Dict[str, Any]:
    """Return configuration merged with user overrides."""

    manager = ConfigManager(
        defaults=deepcopy(defaults or DEFAULT_CONFIG),
        project_root=_resolve_project_root(),
    )
    return manager.load(override_paths=override_paths, include_sources=include_sources)


CONFIG, CONFIG_SOURCES = load_config(include_sources=True)


__all__ = [
    "CONFIG",
    "CONFIG_SOURCES",
    "DEFAULT_CONFIG",
    "ConfigManager",
    "load_config",
]
