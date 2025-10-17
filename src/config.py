"""Minimal configuration helpers for LLMPlotBot."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping

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
    "LM_STUDIO_URL": "http://localhost:1234/v1/chat/completions",
    "MODEL": "creative-writing-model",
    "TITLES_PATH": "data/titles_index.json",
    "PROMPT_PATH": "data/prompt.txt",
    "TEST_SAMPLE_SIZE": 10,
}


@dataclass(frozen=True)
class Settings:
    """Structured settings consumed by the main runner."""

    generated_dir: Path
    request_timeout: int
    write_strategy: str
    write_batch_size: int
    write_batch_seconds: float
    write_batch_retry_limit: int
    file_lock_timeout: float
    file_lock_poll_interval: float
    file_lock_stale_seconds: float
    lm_studio_url: str
    model: str
    titles_path: Path
    prompt_path: Path
    test_sample_size: int

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, base_dir: Path | None = None
    ) -> "Settings":
        base = Path(base_dir or Path.cwd())

        def _resolve_path(value: str | Path) -> Path:
            path = Path(value)
            if not path.is_absolute():
                path = base / path
            return path

        return cls(
            generated_dir=_resolve_path(data["GENERATED_DIR"]),
            request_timeout=int(data["REQUEST_TIMEOUT"]),
            write_strategy=str(data["WRITE_STRATEGY"]),
            write_batch_size=int(data["WRITE_BATCH_SIZE"]),
            write_batch_seconds=float(data["WRITE_BATCH_SECONDS"]),
            write_batch_retry_limit=int(data["WRITE_BATCH_RETRY_LIMIT"]),
            file_lock_timeout=float(data["FILE_LOCK_TIMEOUT"]),
            file_lock_poll_interval=float(data["FILE_LOCK_POLL_INTERVAL"]),
            file_lock_stale_seconds=float(data["FILE_LOCK_STALE_SECONDS"]),
            lm_studio_url=str(data["LM_STUDIO_URL"]),
            model=str(data["MODEL"]),
            titles_path=_resolve_path(data["TITLES_PATH"]),
            prompt_path=_resolve_path(data["PROMPT_PATH"]),
            test_sample_size=int(data["TEST_SAMPLE_SIZE"]),
        )


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load overrides from JSON and merge with :data:`DEFAULT_CONFIG`."""

    config = dict(DEFAULT_CONFIG)
    if path is None:
        path = Path("config") / "config.json"
    else:
        path = Path(path)

    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):  # pragma: no cover - defensive
            raise ValueError("Configuration file must contain a JSON object.")
        config.update(data)

    return config


def load_settings(path: str | Path | None = None) -> Settings:
    """Load :class:`Settings`, resolving relative paths to the config location."""

    if path is None:
        config_path = Path("config") / "config.json"
    else:
        config_path = Path(path)

    data = load_config(config_path)
    base_dir = config_path.parent if config_path.exists() else Path.cwd()
    return Settings.from_mapping(data, base_dir=base_dir)


__all__ = ["DEFAULT_CONFIG", "Settings", "load_config", "load_settings"]
