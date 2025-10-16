"""Utilities for discovering active LM Studio models."""

from __future__ import annotations

import json
import re
import subprocess
from collections import OrderedDict
from typing import Iterable, List, Optional


MODEL_INSTANCE_SUFFIX = re.compile(r":\d+$")


def normalize_model_key(key: str) -> str:
    """Return the base model identifier without LM Studio instance suffixes."""

    if not isinstance(key, str):
        return ""
    return MODEL_INSTANCE_SUFFIX.sub("", key.strip())


def group_model_keys(keys: Iterable[str]) -> "OrderedDict[str, List[str]]":
    """Group LM Studio model keys by their normalized base name."""

    grouped: "OrderedDict[str, List[str]]" = OrderedDict()
    for key in keys:
        base = normalize_model_key(key)
        if not base:
            continue
        grouped.setdefault(base, []).append(key)
    return grouped


def get_model_keys(logger: Optional[object] = None) -> List[str]:
    """Return the model keys for currently running chat-capable LM Studio models."""

    try:
        result = subprocess.run(
            ["lms", "ps", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:  # pragma: no cover - best effort discovery
        if logger:
            logger.error(f"Unable to query LM Studio models: {exc}")
        else:
            print("Error:", exc)
        return []

    try:
        models = json.loads(result.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - guard against malformed output
        if logger:
            logger.error(f"Failed to decode LM Studio response: {exc}")
        else:
            print("Error:", exc)
        return []

    keys: List[str] = []
    for model in models:
        key = model.get("modelKey")
        if not key:
            continue

        lowered = key.lower()
        if "embed" in lowered or "embedding" in lowered:
            continue

        if key not in keys:
            keys.append(key)

    return keys


if __name__ == "__main__":  # pragma: no cover - manual utility execution
    print(get_model_keys())

