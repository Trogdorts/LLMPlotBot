"""Path helpers for consistent, human-readable logging."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    try:
        path = Path(value).expanduser()
    except (TypeError, ValueError):
        return None
    return path


def normalize_for_logging(
    path: Optional[str],
    *,
    base: Optional[str] = None,
    extra_roots: Optional[Iterable[str]] = None,
) -> str:
    """Return a normalised path suited for log output."""

    if path is None:
        return ""

    candidate = _as_path(path)
    if candidate is None:
        return str(path)

    try:
        resolved = candidate.resolve(strict=False)
    except Exception:  # pragma: no cover - extremely defensive
        resolved = Path(os.path.normpath(str(candidate)))

    anchors = []
    base_path = _as_path(base)
    if base_path is not None:
        anchors.append(base_path)
    if extra_roots:
        for root in extra_roots:
            root_path = _as_path(root)
            if root_path is not None:
                anchors.append(root_path)
    anchors.append(PROJECT_ROOT)

    for anchor in anchors:
        try:
            relative = resolved.relative_to(anchor.resolve(strict=False))
            return relative.as_posix()
        except Exception:
            continue

    return resolved.as_posix()


__all__ = ["normalize_for_logging", "PROJECT_ROOT"]
