from __future__ import annotations

import json
from pathlib import Path

from src.config import DEFAULT_CONFIG, Settings, load_config, load_settings


def test_load_config_merges_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    overrides = {"REQUEST_TIMEOUT": 15, "LM_STUDIO_URL": "http://test"}
    config_path.write_text(json.dumps(overrides), encoding="utf-8")

    data = load_config(config_path)

    assert data["REQUEST_TIMEOUT"] == 15
    assert data["LM_STUDIO_URL"] == "http://test"
    assert data["GENERATED_DIR"] == DEFAULT_CONFIG["GENERATED_DIR"]


def test_load_settings_resolves_relative_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_path = config_dir / "settings.json"
    overrides = {
        "GENERATED_DIR": "./output",
        "TITLES_PATH": "./titles.json",
        "PROMPT_PATH": "./prompt.txt",
    }
    config_path.write_text(json.dumps(overrides), encoding="utf-8")

    settings = load_settings(config_path)

    assert isinstance(settings, Settings)
    assert settings.generated_dir == config_dir / "output"
    assert settings.titles_path == config_dir / "titles.json"
    assert settings.prompt_path == config_dir / "prompt.txt"
